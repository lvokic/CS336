#include <pybind11/pybind11.h>
#include <algorithm>
#include <array>
#include <queue>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "training.hpp"
#include "pair_queue.hpp"

namespace py = pybind11;

[[noreturn]] void unfinished(const char* message) {
    PyErr_SetString(PyExc_NotImplementedError, message);
    throw py::error_already_set();
}

bpe::TrainingState prepare_training(
    const py::dict& pretoken_counts, int vocab_size, const py::list& special_tokens
) {
    if (vocab_size < 256) {
        throw py::value_error("vocab_size must be at least 256");
    }
    if (pretoken_counts.size() > std::numeric_limits<bpe::WordId>::max()) {
        throw py::value_error("Too many distinct pretokens for WordId");
    }

    bpe::TrainingState state;
    state.token_bytes.reserve(256);
    for (unsigned int byte = 0; byte < 256; ++byte) {
        state.token_bytes.emplace_back(1, static_cast<char>(byte));
    }

    std::unordered_set<std::string> seen_specials;
    for (py::handle item : special_tokens) {
        if (!py::isinstance<py::str>(item)) {
            throw py::type_error("special_tokens must contain str values");
        }
        auto bytes = py::cast<std::string>(item);  // str -> UTF-8 bytes
        if (bytes.empty()) {
            throw py::value_error("Special tokens must not be empty");
        }
        if (seen_specials.insert(bytes).second) {
            if (state.token_bytes.size() >= static_cast<std::size_t>(vocab_size)) {
                throw py::value_error("vocab_size cannot accommodate the special tokens");
            }
            state.token_bytes.push_back(std::move(bytes));
        }
    }

    state.words.reserve(pretoken_counts.size());
    for (auto item : pretoken_counts) {
        if (!py::isinstance<py::bytes>(item.first)) {
            throw py::type_error("pretoken_counts keys must be bytes");
        }
        if (!py::isinstance<py::int_>(item.second) || PyBool_Check(item.second.ptr())) {
            throw py::type_error("pretoken_counts values must be integer counts");
        }
        const auto frequency = PyLong_AsLongLong(item.second.ptr());
        if (frequency == -1 && PyErr_Occurred()) {
            throw py::error_already_set();
        }
        if (frequency <= 0) {
            throw py::value_error("Pretoken counts must be positive");
        }
        const auto bytes = py::cast<std::string>(item.first);
        if (bytes.empty()) {
            throw py::value_error("Pretokens must not be empty");
        }
        state.words.push_back(bpe::make_word(bytes, frequency));
    }
    return state;
}

py::tuple train_bpe(py::dict pretoken_counts, int vocab_size, py::list special_tokens) {
    auto state = prepare_training(pretoken_counts, vocab_size, special_tokens);
    if (state.token_bytes.size() == static_cast<std::size_t>(vocab_size)) {
        py::dict vocab;
        for (std::size_t id = 0; id < state.token_bytes.size(); ++id) {
            vocab[py::int_(id)] = py::bytes(state.token_bytes[id]);
        }
        return py::make_tuple(vocab, py::list());
    }
    std::vector<std::pair<std::string, std::string>> merges;
    merges.reserve(static_cast<std::size_t>(vocab_size) - state.token_bytes.size());
    {
        py::gil_scoped_release release;
        const auto hardware_threads = std::thread::hardware_concurrency();
        const auto worker_count = std::min<unsigned>(
            hardware_threads == 0 ? 1u : hardware_threads, 32u
        );
        bpe::initialize_statistics_parallel(state, worker_count);
        bpe::PairQueue queue(state);
        bpe::UpdateScratch scratch;

        while (state.token_bytes.size() < static_cast<std::size_t>(vocab_size)) {
            const auto best = queue.best();
            if (!best) {
                break;
            }
            const auto left = bpe::pair_left(best->pair);
            const auto right = bpe::pair_right(best->pair);
            const auto merged = static_cast<bpe::TokenId>(state.token_bytes.size());

            // Copy the membership set before replace_word mutates the index.
            const auto members_it = state.pair_words.find(best->pair);
            if (members_it == state.pair_words.end()) {
                throw std::logic_error("Selected pair has no inverted-index members");
            }
            const auto members = members_it->second;
            const auto left_bytes = state.token_bytes.at(left);
            const auto right_bytes = state.token_bytes.at(right);
            std::string merged_bytes = left_bytes;
            merged_bytes += right_bytes;

            state.token_bytes.push_back(merged_bytes);
            merges.emplace_back(left_bytes, right_bytes);

            std::unordered_set<bpe::PairKey> changed_pairs;
            for (const auto word_id : members) {
                auto replacement = bpe::merge_pair(state.words.at(word_id).tokens, left, right, merged);
                bpe::replace_word(state, word_id, std::move(replacement), scratch, changed_pairs);
            }
            queue.publish(changed_pairs);
        }
    }

    py::dict vocab;
    for (std::size_t id = 0; id < state.token_bytes.size(); ++id) {
        vocab[py::int_(id)] = py::bytes(state.token_bytes[id]);
    }
    py::list merge_list;
    for (const auto& [left, right] : merges) {
        merge_list.append(py::make_tuple(py::bytes(left), py::bytes(right)));
    }
    return py::make_tuple(vocab, merge_list);
}

class TokenizerCore {
public:
    TokenizerCore(py::dict vocab, py::list merges, py::list special_tokens)
        : merges_(merges), special_tokens_(special_tokens) {
        byte_token_ids_.fill(std::numeric_limits<bpe::TokenId>::max());
        std::size_t max_id = 0;
        bool has_entries = false;
        std::unordered_map<std::string, bpe::TokenId> token_ids;
        for (auto item : vocab) {
            if (!py::isinstance<py::int_>(item.first) || PyBool_Check(item.first.ptr())) {
                throw py::type_error("vocab keys must be integer token IDs");
            }
            const auto id = PyLong_AsLongLong(item.first.ptr());
            if (id < 0 || (id == -1 && PyErr_Occurred())) {
                if (PyErr_Occurred()) { throw py::error_already_set(); }
                throw py::value_error("vocab token IDs must be non-negative");
            }
            if (!py::isinstance<py::bytes>(item.second)) {
                throw py::type_error("vocab values must be bytes");
            }
            max_id = std::max(max_id, static_cast<std::size_t>(id));
            has_entries = true;
        }
        if (has_entries) {
            vocab_bytes_.resize(max_id + 1);
            for (auto item : vocab) {
                const auto id = static_cast<std::size_t>(PyLong_AsLongLong(item.first.ptr()));
                vocab_bytes_[id] = py::cast<std::string>(item.second);
                token_ids[vocab_bytes_[id]] = static_cast<bpe::TokenId>(id);
                if (vocab_bytes_[id].size() == 1) {
                    byte_token_ids_[static_cast<unsigned char>(vocab_bytes_[id][0])] =
                        static_cast<bpe::TokenId>(id);
                }
            }
        }
        merge_ranks_.reserve(merges.size());
        merge_results_.reserve(merges.size());
        for (std::size_t rank = 0; rank < merges.size(); ++rank) {
            const auto merge = py::cast<py::tuple>(merges[rank]);
            if (merge.size() != 2 || !py::isinstance<py::bytes>(merge[0]) ||
                !py::isinstance<py::bytes>(merge[1])) {
                throw py::type_error("merges must contain pairs of bytes");
            }
            const auto left_bytes = py::cast<std::string>(merge[0]);
            const auto right_bytes = py::cast<std::string>(merge[1]);
            const auto left = token_ids.find(left_bytes);
            const auto right = token_ids.find(right_bytes);
            std::string merged_bytes = left_bytes + right_bytes;
            const auto result = token_ids.find(merged_bytes);
            if (left == token_ids.end() || right == token_ids.end() || result == token_ids.end()) {
                throw py::value_error("merge tokens must exist in vocab, including their concatenation");
            }
            const auto key = bpe::pack_pair(left->second, right->second);
            if (!merge_ranks_.emplace(key, static_cast<std::uint32_t>(rank)).second) {
                throw py::value_error("duplicate merge pair");
            }
            merge_results_[key] = result->second;
        }
    }

    py::list encode_pretoken(py::bytes pretoken) const {
        const auto bytes = py::cast<std::string>(pretoken);
        std::vector<bpe::TokenId> encoded;
        {
            py::gil_scoped_release release;
            encoded = encode_bytes(bytes);
        }
        py::list result;
        for (const auto id : encoded) {
            result.append(py::int_(id));
        }
        return result;
    }

    py::bytes decode_bytes(py::list ids) const {
        std::vector<bpe::TokenId> token_ids;
        token_ids.reserve(ids.size());
        for (py::handle item : ids) {
            if (!py::isinstance<py::int_>(item) || PyBool_Check(item.ptr())) {
                throw py::type_error("token IDs must be integers");
            }
            const auto id = PyLong_AsLongLong(item.ptr());
            if (id == -1 && PyErr_Occurred()) {
                throw py::error_already_set();
            }
            if (id < 0 || static_cast<std::uint64_t>(id) >= vocab_bytes_.size()) {
                throw py::index_error("token ID is not present in the vocabulary");
            }
            token_ids.push_back(static_cast<bpe::TokenId>(id));
        }

        std::size_t total_size = 0;
        for (const auto id : token_ids) {
            const auto& bytes = vocab_bytes_[id];
            if (bytes.size() > std::numeric_limits<std::size_t>::max() - total_size) {
                throw std::overflow_error("decoded bytes exceed addressable size");
            }
            total_size += bytes.size();
        }
        std::string decoded;
        {
            py::gil_scoped_release release;
            decoded.reserve(total_size);
            for (const auto id : token_ids) {
                decoded += vocab_bytes_[id];
            }
        }
        return py::bytes(decoded);
    }

private:
    struct Node {
        bpe::TokenId token;
        std::size_t previous;
        std::size_t next;
        bool alive;
    };

    struct Candidate {
        std::uint32_t rank;
        std::size_t position;
        bpe::TokenId left;
        bpe::TokenId right;
    };

    struct CandidateGreater {
        bool operator()(const Candidate& a, const Candidate& b) const {
            if (a.rank != b.rank) { return a.rank > b.rank; }
            return a.position > b.position;
        }
    };

    std::vector<bpe::TokenId> encode_bytes(const std::string& bytes) const {
        constexpr auto none = std::numeric_limits<std::size_t>::max();
        if (bytes.empty()) { return {}; }

        std::vector<Node> nodes;
        nodes.reserve(bytes.size());
        for (std::size_t i = 0; i < bytes.size(); ++i) {
            const auto byte_id = byte_token_ids_[static_cast<unsigned char>(bytes[i])];
            if (byte_id == std::numeric_limits<bpe::TokenId>::max()) {
                throw std::invalid_argument("vocab is missing a byte token");
            }
            nodes.push_back({byte_id,
                             i == 0 ? none : i - 1,
                             i + 1 == bytes.size() ? none : i + 1, true});
        }

        std::priority_queue<Candidate, std::vector<Candidate>, CandidateGreater> pending;
        auto add_candidate = [&](std::size_t position) {
            if (position == none || !nodes[position].alive || nodes[position].next == none) {
                return;
            }
            const auto right = nodes[position].next;
            const auto key = bpe::pack_pair(nodes[position].token, nodes[right].token);
            const auto rank = merge_ranks_.find(key);
            if (rank != merge_ranks_.end()) {
                pending.push({rank->second, position, nodes[position].token, nodes[right].token});
            }
        };
        for (std::size_t i = 0; i + 1 < nodes.size(); ++i) { add_candidate(i); }

        while (!pending.empty()) {
            const auto candidate = pending.top();
            pending.pop();
            if (!nodes[candidate.position].alive ||
                nodes[candidate.position].next == none) {
                continue;
            }
            const auto right = nodes[candidate.position].next;
            if (!nodes[right].alive || nodes[candidate.position].token != candidate.left ||
                nodes[right].token != candidate.right) {
                continue;
            }
            const auto key = bpe::pack_pair(candidate.left, candidate.right);
            const auto result = merge_results_.find(key);
            if (result == merge_results_.end()) { continue; }

            const auto previous = nodes[candidate.position].previous;
            const auto next = nodes[right].next;
            nodes[candidate.position].token = result->second;
            nodes[candidate.position].next = next;
            nodes[right].alive = false;
            if (next != none) { nodes[next].previous = candidate.position; }
            add_candidate(previous);
            add_candidate(candidate.position);
        }

        std::vector<bpe::TokenId> result;
        result.reserve(nodes.size());
        for (std::size_t position = 0; position != none; position = nodes[position].next) {
            if (nodes[position].alive) { result.push_back(nodes[position].token); }
        }
        return result;
    }

    py::list merges_;
    py::list special_tokens_;
    std::vector<std::string> vocab_bytes_;
    std::array<bpe::TokenId, 256> byte_token_ids_;
    std::unordered_map<bpe::PairKey, std::uint32_t> merge_ranks_;
    std::unordered_map<bpe::PairKey, bpe::TokenId> merge_results_;
};

PYBIND11_MODULE(_cs336_bpe, m) {
    m.doc() = "CS336 BPE scaffold: interfaces only; algorithms are unfinished.";
    m.def("train_bpe", &train_bpe, py::arg("pretoken_counts"),
          py::arg("vocab_size"), py::arg("special_tokens"));
    py::class_<TokenizerCore>(m, "TokenizerCore")
        .def(py::init<py::dict, py::list, py::list>(), py::arg("vocab"),
             py::arg("merges"), py::arg("special_tokens"))
        .def("encode_pretoken", &TokenizerCore::encode_pretoken, py::arg("pretoken"))
        .def("decode_bytes", &TokenizerCore::decode_bytes, py::arg("ids"));
}
