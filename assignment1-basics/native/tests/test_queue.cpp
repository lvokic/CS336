#include "../src/pair_queue.hpp"
#include <cassert>
#include <random>
#include <tuple>

using Bytes = std::vector<unsigned char>;
Bytes bytes(const std::string& value) { return Bytes(value.begin(), value.end()); }

void check(bpe::PairQueue& queue, const bpe::TrainingState& state) {
    using Rank = std::tuple<bpe::Count, Bytes, Bytes, bpe::PairKey>;
    std::optional<Rank> expected;
    for (const auto& [pair, count] : state.pair_counts) {
        if (count <= 0) { continue; }
        const Rank rank{count, bytes(state.token_bytes.at(bpe::pair_left(pair))),
                        bytes(state.token_bytes.at(bpe::pair_right(pair))), pair};
        if (!expected || rank > *expected) { expected = rank; }
    }
    const auto actual = queue.best();
    assert(actual.has_value() == expected.has_value());
    if (actual) {
        assert(actual->pair == std::get<3>(*expected));
        assert(actual->count == std::get<0>(*expected));
        assert(queue.best()->version == actual->version);
    }
}

int main() {
    bpe::TrainingState state;
    // IDs deliberately disagree with lexical order; include identical
    // concatenations, prefixes, NUL and non-ASCII bytes.
    state.token_bytes = {"z", "a", "ab", "bc", "c", std::string("\x80", 1),
                         std::string("\xff", 1), std::string("\0", 1)};
    state.words = {{{1, 3}, 5}, {{2, 4}, 5}, {{0, 1}, 1}};
    bpe::initialize_statistics(state);
    bpe::PairQueue queue(state);
    check(queue, state);
    assert(queue.best()->pair == bpe::pack_pair(2, 4)); // (ab,c) > (a,bc)
    bpe::UpdateScratch scratch;
    std::unordered_set<bpe::PairKey> changed;
    bpe::replace_word(state, 0, {6, 7}, scratch, changed);
    bpe::replace_word(state, 1, {5, 7}, scratch, changed);
    queue.publish(changed);
    check(queue, state);
    assert(queue.best()->pair == bpe::pack_pair(6, 7));

    // Remove and reintroduce the same pair with exactly the same frequency.
    const auto old_version = queue.best()->version;
    changed.clear();
    bpe::replace_word(state, 0, {1}, scratch, changed);
    queue.publish(changed);
    changed.clear();
    bpe::replace_word(state, 0, {6, 7}, scratch, changed);
    queue.publish(changed);
    assert(queue.best()->version != old_version);
    check(queue, state);

    // Vocabulary growth can reallocate its backing array without invalidating
    // the queue's reference to the vector object.
    for (int i = 0; i < 300; ++i) { state.token_bytes.push_back("extra" + std::to_string(i)); }
    changed.clear();
    bpe::replace_word(state, 2, {300, 1, 300, 1}, scratch, changed);
    queue.publish(changed);
    check(queue, state);

    std::mt19937 rng(337);
    for (int round = 0; round < 1000; ++round) {
        changed.clear();
        for (int update = 0; update < 3; ++update) {
            std::vector<bpe::TokenId> tokens(rng() % 12);
            for (auto& token : tokens) { token = rng() % 8; }
            bpe::replace_word(state, rng() % state.words.size(), std::move(tokens), scratch, changed);
        }
        queue.publish(changed);
        check(queue, state);
        assert(queue.storage_size() <= 4 * state.pair_counts.size() + 67);
    }
    changed.clear();
    for (std::size_t id = 0; id < state.words.size(); ++id) {
        bpe::replace_word(state, id, {1}, scratch, changed);
    }
    queue.publish(changed);
    assert(!queue.best());
    assert(queue.storage_size() == 0);

    // Keep a fixed winner while many obsolete low-priority entries accumulate.
    // This exercises compaction, not just popping stale entries at the top.
    bpe::TrainingState compact;
    compact.token_bytes = {"a", "b", "c"};
    compact.words = {{{2, 2}, 100}, {{0, 1}, 1}};
    bpe::initialize_statistics(compact);
    bpe::PairQueue compact_queue(compact);
    bool compacted = false;
    for (int round = 0; round < 500; ++round) {
        const auto before = compact_queue.storage_size();
        changed.clear();
        bpe::replace_word(compact, 1, round % 2 ? std::vector<bpe::TokenId>{0, 1}
                                              : std::vector<bpe::TokenId>{1, 0}, scratch, changed);
        compact_queue.publish(changed);
        compacted |= compact_queue.storage_size() < before;
        check(compact_queue, compact);
    }
    assert(compacted);
}
