#pragma once

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace bpe {

using TokenId = std::uint32_t;
using WordId = std::uint32_t;
using Count = std::int64_t;
using PairKey = std::uint64_t;
using PairCounts = std::unordered_map<PairKey, Count>;
using PairWords = std::unordered_map<PairKey, std::unordered_set<WordId>>;

constexpr PairKey pack_pair(TokenId left, TokenId right) noexcept {
    return (PairKey{left} << 32) | PairKey{right};
}

constexpr TokenId pair_left(PairKey pair) noexcept {
    return static_cast<TokenId>(pair >> 32);
}

constexpr TokenId pair_right(PairKey pair) noexcept {
    return static_cast<TokenId>(pair);
}

struct Word {
    std::vector<TokenId> tokens;
    Count frequency;
};

struct TrainingState {
    std::vector<Word> words;
    std::vector<std::string> token_bytes;
    PairCounts pair_counts;
    PairWords pair_words;
    bool statistics_initialized = false;
};

// Reused across word updates. clear() retains bucket capacity, although hash
// nodes may still be allocated again; this is not an allocation-free buffer.
struct UpdateScratch {
    PairCounts old_counts;
    PairCounts new_counts;
};

inline std::vector<TokenId> merge_pair(
    const std::vector<TokenId>& tokens, TokenId left, TokenId right, TokenId merged
) {
    std::vector<TokenId> result;
    result.reserve(tokens.size());
    for (std::size_t i = 0; i < tokens.size();) {
        if (i + 1 < tokens.size() && tokens[i] == left && tokens[i + 1] == right) {
            result.push_back(merged);
            i += 2;
        } else {
            result.push_back(tokens[i]);
            ++i;
        }
    }
    return result;
}

inline Word make_word(const std::string& bytes, Count frequency) {
    Word word{{}, frequency};
    word.tokens.reserve(bytes.size());
    for (unsigned char byte : bytes) {
        word.tokens.push_back(static_cast<TokenId>(byte));
    }
    return word;
}

inline void add_count(PairCounts& counts, PairKey pair, Count amount) {
    auto [entry, inserted] = counts.try_emplace(pair, amount);
    if (!inserted) {
        if (entry->second > std::numeric_limits<Count>::max() - amount) {
            throw std::overflow_error("Pair frequency exceeds int64 range");
        }
        entry->second += amount;
    }
}

// One initialization scan; the merge loop must use replace_word instead.
inline void initialize_statistics(TrainingState& state) {
    if (state.statistics_initialized) {
        throw std::logic_error("Pair statistics already initialized");
    }
    if (state.words.size() > std::numeric_limits<WordId>::max()) {
        throw std::invalid_argument("Too many distinct pretokens for WordId");
    }
    state.pair_counts.clear();
    state.pair_words.clear();
    for (std::size_t id = 0; id < state.words.size(); ++id) {
        const auto& word = state.words[id];
        if (word.frequency <= 0) {
            throw std::invalid_argument("Pretoken counts must be positive");
        }
        for (std::size_t i = 1; i < word.tokens.size(); ++i) {
            const auto pair = pack_pair(word.tokens[i - 1], word.tokens[i]);
            add_count(state.pair_counts, pair, word.frequency);
            state.pair_words[pair].insert(static_cast<WordId>(id));
        }
    }
    state.statistics_initialized = true;
}

inline void initialize_statistics_parallel(TrainingState& state, std::size_t worker_count) {
    if (state.statistics_initialized) {
        throw std::logic_error("Pair statistics already initialized");
    }
    if (state.words.size() > std::numeric_limits<WordId>::max()) {
        throw std::invalid_argument("Too many distinct pretokens for WordId");
    }
    if (worker_count <= 1 || state.words.size() < 2) {
        initialize_statistics(state);
        return;
    }
    worker_count = std::min(worker_count, state.words.size());

    struct LocalStatistics {
        PairCounts counts;
        PairWords words;
    };
    std::vector<LocalStatistics> local(worker_count);
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    std::exception_ptr failure;
    std::mutex failure_mutex;
    const auto worker = [&](std::size_t worker_id) {
        try {
            const auto begin = state.words.size() * worker_id / worker_count;
            const auto end = state.words.size() * (worker_id + 1) / worker_count;
            auto& result = local[worker_id];
            for (std::size_t id = begin; id < end; ++id) {
                const auto& word = state.words[id];
                if (word.frequency <= 0) {
                    throw std::invalid_argument("Pretoken counts must be positive");
                }
                for (std::size_t i = 1; i < word.tokens.size(); ++i) {
                    const auto pair = pack_pair(word.tokens[i - 1], word.tokens[i]);
                    add_count(result.counts, pair, word.frequency);
                    result.words[pair].insert(static_cast<WordId>(id));
                }
            }
        } catch (...) {
            std::lock_guard lock(failure_mutex);
            if (!failure) { failure = std::current_exception(); }
        }
    };
    for (std::size_t worker_id = 0; worker_id < worker_count; ++worker_id) {
        workers.emplace_back(worker, worker_id);
    }
    for (auto& thread : workers) { thread.join(); }
    if (failure) { std::rethrow_exception(failure); }

    state.pair_counts.clear();
    state.pair_words.clear();
    for (auto& result : local) {
        for (const auto& [pair, count] : result.counts) {
            add_count(state.pair_counts, pair, count);
        }
        for (auto& [pair, word_ids] : result.words) {
            auto& members = state.pair_words[pair];
            members.insert(word_ids.begin(), word_ids.end());
        }
    }
    state.statistics_initialized = true;
}

inline void count_word_into(const std::vector<TokenId>& tokens, Count frequency, PairCounts& out) {
    out.clear();
    for (std::size_t i = 1; i < tokens.size(); ++i) {
        add_count(out, pack_pair(tokens[i - 1], tokens[i]), frequency);
    }
}

inline void replace_word(
    TrainingState& state, WordId id, std::vector<TokenId> replacement,
    UpdateScratch& scratch, std::unordered_set<PairKey>& changed_pairs
) {
    if (!state.statistics_initialized) {
        throw std::logic_error("Initialize pair statistics before updating words");
    }
    auto& word = state.words.at(id);
    if (word.tokens == replacement) {
        return;
    }
    count_word_into(word.tokens, word.frequency, scratch.old_counts);
    count_word_into(replacement, word.frequency, scratch.new_counts);

    // Check global additions after subtracting this word's old contribution.
    for (const auto& [pair, new_count] : scratch.new_counts) {
        const auto old = scratch.old_counts.find(pair);
        const Count old_count = old == scratch.old_counts.end() ? 0 : old->second;
        const auto global = state.pair_counts.find(pair);
        const Count global_count = global == state.pair_counts.end() ? 0 : global->second;
        if (global_count - old_count > std::numeric_limits<Count>::max() - new_count) {
            throw std::overflow_error("Pair frequency exceeds int64 range");
        }
    }

    for (const auto& [pair, old_count] : scratch.old_counts) {
        const auto next = scratch.new_counts.find(pair);
        const Count new_count = next == scratch.new_counts.end() ? 0 : next->second;
        if (old_count != new_count) {
            changed_pairs.insert(pair);
            auto global = state.pair_counts.find(pair);
            global->second = global->second - old_count + new_count;
            if (global->second == 0) {
                state.pair_counts.erase(global);
            }
        }
        if (new_count == 0) {
            auto members = state.pair_words.find(pair);
            members->second.erase(id);
            if (members->second.empty()) {
                state.pair_words.erase(members);
            }
        }
    }
    for (const auto& [pair, new_count] : scratch.new_counts) {
        if (scratch.old_counts.find(pair) == scratch.old_counts.end()) {
            changed_pairs.insert(pair);
            add_count(state.pair_counts, pair, new_count);
            state.pair_words[pair].insert(id);
        }
    }
    word.tokens = std::move(replacement);
}

}  // namespace bpe
