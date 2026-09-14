#pragma once

#include "training.hpp"
#include <algorithm>
#include <cstring>
#include <optional>

namespace bpe {

struct HeapEntry {
    Count count;
    PairKey pair;
    std::uint64_t version;
};

inline int compare_bytes(const std::string& a, const std::string& b) {
    const int prefix = std::memcmp(a.data(), b.data(), std::min(a.size(), b.size()));
    if (prefix != 0) { return prefix; }
    return (a.size() > b.size()) - (a.size() < b.size());
}

struct PairPriority {
    const std::vector<std::string>* vocabulary;

    bool operator()(const HeapEntry& a, const HeapEntry& b) const {
        if (a.count != b.count) { return a.count < b.count; }
        const auto& vocab = *vocabulary;
        const int left = compare_bytes(vocab.at(pair_left(a.pair)), vocab.at(pair_left(b.pair)));
        if (left != 0) { return left < 0; }
        const int right = compare_bytes(vocab.at(pair_right(a.pair)), vocab.at(pair_right(b.pair)));
        if (right != 0) { return right < 0; }
        // Only relevant if different IDs have identical byte representations.
        return a.pair < b.pair;
    }
};

// state must outlive the queue and remain at a stable address. Append-only
// vocabulary growth is safe; never change the bytes of an existing token.
class PairQueue {
public:
    explicit PairQueue(const TrainingState& state)
        : state_(state), priority_{&state.token_bytes} {
        if (!state.statistics_initialized) {
            throw std::logic_error("Initialize statistics before constructing the pair queue");
        }
        rebuild();
    }
    PairQueue(const PairQueue&) = delete;
    PairQueue& operator=(const PairQueue&) = delete;

    // Call once after all word updates in a round, before inspecting best().
    void publish(const std::unordered_set<PairKey>& changed_pairs) {
        for (PairKey pair : changed_pairs) {
            const auto current = state_.pair_counts.find(pair);
            if (current == state_.pair_counts.end() || current->second <= 0) {
                versions_.erase(pair);
                continue;
            }
            if (next_version_ == std::numeric_limits<std::uint64_t>::max()) {
                rebuild();
                return;
            }
            const auto version = ++next_version_;
            versions_[pair] = version;
            heap_.push_back({current->second, pair, version});
            std::push_heap(heap_.begin(), heap_.end(), priority_);
        }
        // Occasionally compact obsolete heap entries, without rescanning words
        // or rebuilding either global statistics table.
        if (heap_.size() > 64 && (heap_.size() - 64) / 4 > versions_.size()) {
            rebuild();
        }
    }

    // Peek rather than consume: repeated selection without mutation is stable.
    std::optional<HeapEntry> best() {
        while (!heap_.empty()) {
            const auto entry = heap_.front();
            const auto version = versions_.find(entry.pair);
            const auto count = state_.pair_counts.find(entry.pair);
            if (version != versions_.end() && version->second == entry.version &&
                count != state_.pair_counts.end() && count->second == entry.count && entry.count > 0) {
                return entry;
            }
            std::pop_heap(heap_.begin(), heap_.end(), priority_);
            heap_.pop_back();
        }
        return std::nullopt;
    }

    std::size_t storage_size() const noexcept { return heap_.size(); }

private:
    void rebuild() {
        std::vector<HeapEntry> fresh;
        fresh.reserve(state_.pair_counts.size());
        versions_.clear();
        next_version_ = 0;
        for (const auto& [pair, count] : state_.pair_counts) {
            if (count <= 0) { continue; }
            // Validate IDs even if make_heap never invokes the comparator.
            state_.token_bytes.at(pair_left(pair));
            state_.token_bytes.at(pair_right(pair));
            const auto version = ++next_version_;
            versions_[pair] = version;
            fresh.push_back({count, pair, version});
        }
        std::make_heap(fresh.begin(), fresh.end(), priority_);
        heap_.swap(fresh);
    }

    const TrainingState& state_;
    PairPriority priority_;
    std::vector<HeapEntry> heap_;
    std::unordered_map<PairKey, std::uint64_t> versions_;
    std::uint64_t next_version_ = 0;
};

}  // namespace bpe
