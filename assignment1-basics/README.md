# CS336 Assignment1: Tiny GPT Text Generation

本项目基于 Stanford CS336 Assignment1 Basics，实现并串联了一个小型 GPT-style 文本生成模型。项目从底层组件开始实现，包括 BPE Tokenizer、Transformer Block、RoPE Attention、SwiGLU、RMSNorm、AdamW Optimizer、Learning Rate Schedule 等，并在此基础上完成了一个 TinyStories 小样本训练与文本生成实验。

本项目的目标不是训练一个真正可用的大语言模型，而是通过一个 tiny training + generation 实验，理解 GPT 类语言模型从文本数据到生成结果的完整流程。

## 项目内容

核心流程如下：

```text
原始文本
-> BPE Tokenizer
-> token ids
-> get_batch 构造 x/y
-> Transformer Language Model
-> Cross Entropy Loss
-> AdamW 更新参数
-> 保存 checkpoint
-> generate.py 加载模型并生成文本
```

主要实现内容：

- BPE tokenizer 训练、编码与解码
- GPT-style Transformer language model
- Multi-head self-attention with RoPE
- RMSNorm
- SwiGLU feed-forward network
- Cross entropy loss
- AdamW optimizer
- Cosine learning rate schedule
- Checkpoint 保存与加载
- TinyStories 小样本训练
- Loss 曲线可视化
- 文本续写生成

## 项目结构

```text
assignment1-basics/
├── cs336_basics/
│   ├── model.py          # Assignment1 核心实现：tokenizer、Transformer、optimizer 等
│   ├── tiny_lm.py        # 将函数式 transformer_lm 包装成可训练的 nn.Module
│   └── __init__.py
├── tests/                # CS336 Assignment1 测试
├── train.py              # TinyStories 小模型训练脚本
├── generate.py           # 文本生成脚本
├── pyproject.toml        # 项目依赖
├── uv.lock
└── README.md
```

训练过程中会生成以下本地实验产物：

```text
runs/
├── tinystories_300/
├── tinystories_1000/
└── tinystories_5000/
```

这些目录不上传，因为其中包含 checkpoint、token ids、loss 文件等实验产物。

## 环境准备

本项目使用 `uv` 管理 Python 环境。

安装依赖：

```powershell
uv sync
```

如果已经存在 `.venv`，可以直接使用项目虚拟环境中的 Python：

```powershell
.\.venv\Scripts\python --version
```

运行测试建议使用 UTF-8 模式，避免 Windows 默认编码导致 GPT-2 merges 文件读取异常：

```powershell
.\.venv\Scripts\python -X utf8 -m pytest
```

## 数据说明

默认情况下，`train.py` 会按以下顺序寻找训练文本：

```text
data/TinyStoriesV2-GPT4-train.txt
tests/fixtures/tinystories_sample_5M.txt
tests/fixtures/tinystories_sample.txt
```

如果没有手动下载完整 TinyStories 数据，脚本会使用项目自带的 TinyStories fixture 小样本。

如需使用完整 TinyStories 数据，可以新建 `data/` 目录，并放入：

```text
data/TinyStoriesV2-GPT4-train.txt
```

## 训练模型

示例：训练 300 steps：

```powershell
.\.venv\Scripts\python train.py --steps 300 --out-dir runs\tinystories_300 --max-chars 1000000
```

训练 1000 steps：

```powershell
.\.venv\Scripts\python train.py --steps 1000 --out-dir runs\tinystories_1000 --max-chars 1000000
```

训练 5000 steps：

```powershell
.\.venv\Scripts\python train.py --steps 5000 --out-dir runs\tinystories_5000 --max-chars 1000000
```

训练完成后，输出目录中会包含：

```text
checkpoint_final.pt      # 最终模型 checkpoint
loss.svg                 # loss 曲线图
losses.csv               # loss 数值记录
tokenizer.pt             # 训练得到的 tokenizer
tinystories_subset.txt   # 本次实际使用的训练文本
train_ids.npy            # 训练 token ids
valid_ids.npy            # 验证 token ids
```

## 文本生成

使用训练好的 checkpoint 进行生成：

```powershell
.\.venv\Scripts\python generate.py --checkpoint runs\tinystories_5000\checkpoint_final.pt --prompt "Once upon a time, there was a little " --max-new-tokens 120 --temperature 0.7 --top-k 20
```

参数说明：

- `--checkpoint`：模型 checkpoint 路径
- `--prompt`：输入提示词
- `--max-new-tokens`：最多生成的新 token 数
- `--temperature`：采样随机性，越低越稳定，越高越随机
- `--top-k`：每一步只从概率最高的 k 个 token 中采样

如果希望使用最保守的贪心解码，可以设置：

```powershell
--temperature 0
```

## 实验结果

本项目进行了三组 tiny training 实验，训练步数分别为：

```text
300 steps
1000 steps
5000 steps
```

随着训练步数增加，loss 曲线整体下降，说明模型逐渐学习到了 TinyStories 文本中的语言模式。

实验观察：

| 训练步数 | 结果观察 |
| --- | --- |
| 300 | loss 开始下降，但生成文本仍较混乱，容易出现重复和提前结束 |
| 1000 | 生成结果开始出现简单故事结构，如人物、动作、故事开头等 |
| 5000 | 输出更接近 TinyStories 风格，但仍存在重复、语义不连贯和 `<\|endoftext\|>` 提前出现等问题 |

本实验说明，即使是一个小规模 Transformer，只要完成 tokenizer、模型、loss、optimizer 和 generation 的完整闭环，也可以观察到语言模型从随机输出逐渐学习文本模式的过程。

## 关键代码说明

### `cs336_basics/model.py`

Assignment1 的核心实现文件，包含：

- `train_bpe` / `Tokenizer`
- `embedding`
- `linear`
- `rmsnorm`
- `swiglu`
- `scaled_dot_product_attention`
- `multihead_self_attention_with_rope`
- `transformer_block`
- `transformer_lm`
- `cross_entropy`
- `get_batch`
- `AdamW`

### `cs336_basics/tiny_lm.py`

将函数式的 `transformer_lm(...)` 包装成 PyTorch 可训练的 `nn.Module`。

原始 Assignment1 实现中，模型前向传播需要手动传入 `weights` 字典。为了训练，需要让 PyTorch 能够管理参数，因此 `tiny_lm.py` 中创建了 `TinyTransformerLM`，将模型权重注册为 `nn.Parameter`，并在 `forward()` 中重新整理成 `transformer_lm` 所需的 `weights` 格式。

### `train.py`

完整训练脚本，负责：

- 读取 TinyStories 文本
- 训练 BPE tokenizer
- 编码文本为 token ids
- 切分训练集和验证集
- 创建 tiny Transformer
- 计算 cross entropy loss
- 使用 AdamW 更新参数
- 保存 checkpoint
- 写出 loss CSV 和 SVG 曲线

### `generate.py`

文本生成脚本，负责：

- 加载 checkpoint
- 还原模型和 tokenizer
- 编码 prompt
- 自回归预测下一个 token
- 使用 temperature / top-k 采样
- decode token ids 为文本

## 注意事项

1. 当前项目环境中的 PyTorch 可能是 CPU-only 版本。如果 `torch.cuda.is_available()` 为 `False`，训练会使用 CPU。
2. 小模型和小数据训练的生成质量有限，结果不能和真正的大语言模型相比。
3. `<|endoftext|>` 是 TinyStories 中的特殊结束符，生成时出现它属于正常现象。
4. `runs/` 和 `data/` 通常不建议上传 GitHub，可通过 `.gitignore` 忽略。

## 参考

- Stanford CS336 Assignment1 Basics
- TinyStories Dataset
- Transformer / GPT-style Language Modeling
