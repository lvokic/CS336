# C++ BPE 脚手架

已实现 C++ 训练输入转换、初始词表、相邻 pair 加权频次统计、倒排索引、最大堆、BPE 合并循环、编码和解码。
目标词表恰好容纳初始字节和特殊 token 时，native 训练入口可以返回零条合并规则。
Python 公共训练入口仍需要实现语料准备与 native 调用。

`train_bpe` 现在支持 `num_workers` 和 `chunk_bytes` 参数。大文件会按 UTF-8 行边界分块，
由多个进程分别构建本地 pretoken Counter，再由主进程归并；文件小于一个 chunk 时自动走串行路径。
默认最多使用 32 个 worker，默认块大小为 64 MiB。训练测试可以用 `num_workers=1` 固定为串行。

训练完成后可使用 `cs336_basics.bpe.save_tokenizer` 保存二进制 tokenizer，
以后用 `load_tokenizer` 直接加载 vocab、merges 和 special tokens，不必重新训练。
格式使用固定 magic、版本号、数量字段和长度前缀 bytes；不使用 JSON，也不依赖 pickle。
保存先写同目录临时文件、fsync 后原子替换目标文件；加载会检查版本、截断和尾随数据。
该格式按 vocab ID 顺序写入，避免保存 Python dict 的键开销；当前实现仍在 Python 层写文件，
训练和编码的热路径不受序列化格式影响。

## 构建与测试

在 assignment1-basics 根目录执行：

```sh
uv pip install --python .venv/bin/python ./native
.venv/bin/python -c 'import _cs336_bpe; print(_cs336_bpe.__doc__)'
uv run --no-sync pytest tests/test_train_bpe.py tests/test_tokenizer.py
```

修改 C++ 后重新执行安装命令。需要 C++17 编译器和对应 Python 开发头文件。
扩展独立于主项目的 uv_build 后端；主项目 uv.lock 没有包含它。
`uv sync` 可能移除单独安装的扩展，之后重新安装即可。
VSCode 选择根目录 `.venv/bin/python`。

## VSCode C++ 头文件提示

构建隔离环境中的 pybind11 不会自动安装到项目环境。供编辑器使用时执行：

```sh
uv pip install --python .venv/bin/python 'pybind11>=3.0,<4'
.venv/bin/python -m pybind11 --includes
```

工作区和作业目录的 `.vscode/c_cpp_properties.json` 已配置当前机器的
编译器、C++17、pybind11 和 Python 头文件目录。迁移机器或更换 Python
后，按上面命令的实际输出更新路径。`uv sync` 后可能需要重新安装 pybind11。
仍有旧诊断时，运行 VSCode 的 `C/C++: Reset IntelliSense Database`。

构建方式参考 [pybind11 官方文档](https://pybind11.readthedocs.io/en/stable/compiling.html)
和 [uv 本地包安装说明](https://docs.astral.sh/uv/pip/packages/)。

## 文件职责

- `cs336_basics/bpe.py`：Python 公共接口；语料读取、Unicode 正则预分词、特殊 token 和流式输入处理的待实现位置。
- `native/src/bindings.cpp`：C++ 接口和绑定；训练、单个 pretoken 编码、ID 到字节转换的待实现位置。
- `tests/adapters.py`：仅转发到公共接口。

`src/training.hpp` 定义纯 C++ 的 `Word` 和 `TrainingState`。
训练入口一次性转换输入，保存整数 token 序列、64 位频次和拥有存储的字节词表。
统计期间释放 GIL；返回 Python 前重新获取。`initialize_statistics_parallel` 将唯一 pretoken
按范围分给 worker，各 worker 使用本地哈希表，最后单线程归并；归并结果与单线程初始化一致。
统计的平均时间复杂度为 O(N)，
N 是所有唯一 pretoken 的相邻位置总数，空间复杂度为 O(P)，P 是不同 pair 数量。
训练入口使用 `initialize_statistics` 一次扫描同时初始化两张表，重复初始化会报错。
重复出现和重叠的 pair 在每个词中只记录一个 WordId，不按词频复制索引。
WordId 是 words 数组下标，后续合并阶段必须保持数组顺序稳定。
平均构建时间为 O(N)，空间为 O(P + M)，M 是不同 (pair, WordId) 关系数。

后续训练循环可复用 `replace_word(state, id, replacement, scratch, changed_pairs)`：
只扫描该词更新前后的 token 序列，原地更新全局频次与词集合，并删除归零/空条目。
`UpdateScratch` 在训练循环外创建一次，重复使用局部表的 bucket 容量；仍可能分配哈希节点。
`changed_pairs` 在每轮开始清空，跨该轮所有受影响词累积，供后续最大堆更新。
词频保持不变，WordId 顺序不能改变；初始化后不要绕过该接口直接修改 words。
算术溢出在全局修改前检查；修改期间若内存分配失败，应终止训练，不保证事务回滚。
局部更新平均耗时取决于该词更新前后的长度，不依赖全语料长度。
训练入口默认使用 `std::thread::hardware_concurrency()` 个 worker；merge loop 仍然单线程，
因为每次 merge 的结果决定下一次 merge。worker 数量为 1 时自动退化为串行初始化。
回归测试 `native/tests/test_updates.cpp` 包含 1000 次连续随机增量更新。

`src/pair_queue.hpp` 的 `PairQueue` 在统计初始化后创建一次，持有 TrainingState 的引用。
训练状态必须比堆活得更久、不能移动；词表允许追加，但不能修改已有 token 的字节内容。
每轮完成全部 `replace_word` 后调用 `queue.publish(changed_pairs)`，再用 `queue.best()`
读取最高优先级 pair。best 是非消耗式读取，空堆返回 nullopt。
排序依次比较频次、左 token 字节、右 token 字节，均选择较大者；字节使用无符号顺序。
堆项记录频次快照和版本号，频次表变化不会直接修改比较器所依赖的排序字段。
publish 为变化的有效 pair 入堆；best 惰性丢弃失效版本，删除再出现的 pair 也获得新版本。
陈旧条目过多时偶尔从现有 pair_counts 压缩重建堆，不扫描 words，也不重建频次或倒排索引。
通常单次更新/弹出为 O(log H)，H 是堆条目数；首次建堆和偶尔压缩为 O(P)，
字典序比较额外取决于所比较 token 的公共前缀长度。
测试 `native/tests/test_queue.cpp` 验证平局规则、二进制字节、版本失效、词表扩容、
1000 轮随机增量更新与全量选择的一致性，以及陈旧堆条目的压缩。

训练循环每轮取出 `queue.best()`，复制该 pair 的 WordId 集合，给左右 token 拼接出新 token，
对每个受影响词执行左到右的非重叠合并，再调用 `replace_word` 和 `queue.publish`。
如果没有可用 pair，会提前结束，即使目标词表还没有填满。

`TokenizerCore::decode_bytes` 构造一次拥有存储的 ID 到 bytes 数组，调用时只需校验 ID、
计算输出长度并顺序拼接；拼接期间释放 GIL，Python 层再把 UTF-8 bytes 解码为 str。
Python `encode_iterable` 维护 pending buffer：只立即处理已经确定不会跨 chunk 的 pretoken，
保留末尾可能不完整的普通 token 或 special token 前缀，迭代结束后再 flush。其内部内存不随输入总长度增长。
空 pretoken、非正频次、超出 int64 的频次会被拒绝；重复特殊 token 按首次出现顺序去重。
TokenizerCore 目前保存 Python 容器只是为了建立可调用接口，不是高性能存储设计。
后续可将核心类型和算法拆分到独立 C++ 文件，并更新 setup.py 的源文件列表。
不要在仍访问 Python 对象时释放 GIL。

输入转换回归测试：`.venv/bin/python -m pytest native/tests/test_input.py`。
纯 C++ 字节转换测试：编译运行 `native/tests/test_training.cpp`（C++17）。

## 接口契约

- native `train_bpe` 输入为 `dict[bytes, int]` 预分词频次、目标词表大小和 `list[str]` 特殊 token。
- 训练返回 `dict[int, bytes]` 词表和按学习顺序排列的 `list[tuple[bytes, bytes]]` merges。
- native `encode_pretoken` 接收单个预分词的 bytes，返回 `list[int]`。
- native `decode_bytes` 接收 `list[int]`，返回 bytes；Python 公共 decode 返回 str。
- 公共 `encode_iterable` 接收文本迭代器，返回 token ID 迭代器。

## 开始顺序

先读 handout §2.1–2.4，再对照训练测试理解训练输出；然后阅读 §2.6 和 tokenizer 测试。
训练负责从语料学习词表与合并规则；编码使用已学习的规则；解码恢复文本。
特殊 token、频次平局、UTF-8 错误处理和流式输入的行为以 handout 和测试为准。
先验证正确性，再单独测量读取、预分词、训练、编码和绑定开销。
