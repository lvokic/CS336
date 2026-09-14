#include "../src/training.hpp"
#include <cassert>
#include <random>

int main() {
    bpe::TrainingState state;
    state.words = {bpe::make_word("banana", 3), bpe::make_word("band", 2),
                   bpe::make_word("aaaa", 5), bpe::make_word("xy", 9)};
    bpe::initialize_statistics(state);
    bool rejected = false;
    try { bpe::initialize_statistics(state); }
    catch (const std::logic_error&) { rejected = true; }
    assert(rejected);

    bpe::UpdateScratch scratch;
    std::unordered_set<bpe::PairKey> changed;
    // An unrelated global entry must survive local updates in place.
    const auto xy = bpe::pack_pair('x', 'y');
    const auto* untouched_count = &state.pair_counts.at(xy);
    const auto* untouched_members = &state.pair_words.at(xy);
    bpe::replace_word(state, 0, {'b', 256, 256, 'a'}, scratch, changed);
    assert(changed.count(xy) == 0);
    assert(&state.pair_counts.at(xy) == untouched_count);
    assert(&state.pair_words.at(xy) == untouched_members);
    assert(state.pair_words.at(bpe::pack_pair('a', 'n')) == std::unordered_set<bpe::WordId>{1});
    bpe::replace_word(state, 1, {'b', 256, 'd'}, scratch, changed);
    assert(state.pair_counts.count(bpe::pack_pair('a', 'n')) == 0);
    assert(state.pair_words.count(bpe::pack_pair('a', 'n')) == 0);
    bpe::replace_word(state, 2, {257, 257}, scratch, changed);
    assert(state.pair_counts.at(bpe::pack_pair(257, 257)) == 5);
    changed.clear();
    bpe::replace_word(state, 2, {257, 257}, scratch, changed);
    assert(changed.empty());

    // Random consecutive replacements exercise disappearance/reappearance,
    // overlap, count increases/decreases, and changing word membership.
    std::mt19937 rng(336);
    for (int step = 0; step < 1000; ++step) {
        const bpe::WordId id = rng() % state.words.size();
        std::vector<bpe::TokenId> tokens(rng() % 15);
        for (auto& token : tokens) { token = rng() % 5; }
        const auto before = state.pair_counts;
        changed.clear();
        bpe::replace_word(state, id, std::move(tokens), scratch, changed);
        std::unordered_set<bpe::PairKey> expected;
        for (const auto& [pair, count] : before) {
            const auto now = state.pair_counts.find(pair);
            if (now == state.pair_counts.end() || now->second != count) { expected.insert(pair); }
        }
        for (const auto& [pair, count] : state.pair_counts) {
            if (before.count(pair) == 0) { expected.insert(pair); }
        }
        assert(changed == expected);
    }

    // Even a cross-word global overflow must leave state and change set intact.
    bpe::TrainingState large;
    large.words = {bpe::make_word("ab", std::numeric_limits<bpe::Count>::max()),
                   bpe::make_word("cd", 1)};
    bpe::initialize_statistics(large);
    changed.clear();
    rejected = false;
    try { bpe::replace_word(large, 1, {'a', 'b'}, scratch, changed); }
    catch (const std::overflow_error&) { rejected = true; }
    assert(rejected);
    assert(changed.empty());
    assert(large.words[1].tokens == (std::vector<bpe::TokenId>{'c', 'd'}));
}
