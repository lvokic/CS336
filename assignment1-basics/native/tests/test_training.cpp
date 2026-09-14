#include "../src/training.hpp"
#include <cassert>

int main() {
    const std::string bytes("\x00\x80\xff", 3);
    const auto word = bpe::make_word(bytes, bpe::Count{1} << 40);
    assert((word.tokens == std::vector<bpe::TokenId>{0, 128, 255}));
    assert(word.frequency == (bpe::Count{1} << 40));

    const auto max_id = std::numeric_limits<bpe::TokenId>::max();
    static_assert(bpe::pack_pair(1, 2) != bpe::pack_pair(2, 1));
    assert(bpe::pair_left(bpe::pack_pair(max_id, 0)) == max_id);
    assert(bpe::pair_right(bpe::pack_pair(0, max_id)) == max_id);
    assert(bpe::pair_left(bpe::pack_pair(256, 10000)) == 256);
    assert(bpe::pair_right(bpe::pack_pair(256, 10000)) == 10000);

    bpe::TrainingState state;
    state.words = {bpe::make_word("banana", 3), bpe::make_word("band", 2),
                   bpe::make_word("aaaa", 5), word, {{256, max_id, 256}, 9}};
    bpe::initialize_statistics(state);
    using WordSet = std::unordered_set<bpe::WordId>;
    assert(state.pair_counts.at(bpe::pack_pair('b', 'a')) == 5);
    assert(state.pair_counts.at(bpe::pack_pair('a', 'n')) == 8);
    assert(state.pair_counts.at(bpe::pack_pair('n', 'a')) == 6);
    assert(state.pair_counts.at(bpe::pack_pair('a', 'a')) == 15);
    assert(state.pair_counts.at(bpe::pack_pair(0, 128)) == word.frequency);
    assert(state.pair_counts.at(bpe::pack_pair(128, 255)) == word.frequency);
    assert(state.pair_counts.at(bpe::pack_pair(256, max_id)) == 9);
    assert(state.pair_counts.at(bpe::pack_pair(max_id, 256)) == 9);
    assert(state.pair_words.at(bpe::pack_pair('b', 'a')) == (WordSet{0, 1}));
    assert(state.pair_words.at(bpe::pack_pair('a', 'n')) == (WordSet{0, 1}));
    assert(state.pair_words.at(bpe::pack_pair('n', 'a')) == (WordSet{0}));
    assert(state.pair_words.at(bpe::pack_pair('a', 'a')) == (WordSet{2}));
    assert(state.pair_words.at(bpe::pack_pair(0, 128)) == (WordSet{3}));
    assert(state.pair_words.at(bpe::pack_pair(256, max_id)) == (WordSet{4}));

    bpe::TrainingState parallel;
    parallel.words = state.words;
    bpe::initialize_statistics_parallel(parallel, 4);
    assert(parallel.pair_counts == state.pair_counts);
    assert(parallel.pair_words == state.pair_words);

    assert((bpe::merge_pair({1, 1, 1, 2}, 1, 1, 256) ==
            std::vector<bpe::TokenId>{256, 1, 2}));
    assert((bpe::merge_pair({1, 2, 1, 2}, 1, 2, 256) ==
            std::vector<bpe::TokenId>{256, 256}));
}
