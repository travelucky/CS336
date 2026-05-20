# CS336 Assignment 1 Basics 学习笔记

这份笔记按你完成作业时的学习顺序整理。重点不是背代码，而是记住每个模块在语言模型里的位置、输入输出、shape 变化、核心公式和容易错的地方。

## 0. 总体路线

Assignment 1 可以理解成三条线：

1. 模型线：从 `linear`、`embedding` 一路搭到完整 `Transformer LM`。
2. 训练线：实现 loss、batch sampling、gradient clipping、AdamW、LR schedule、checkpoint。
3. Tokenizer 线：实现 BPE tokenizer 的 encode/decode，以及从语料训练 BPE merges。

完整训练流程大概是：

```text
raw text
-> tokenizer.encode
-> token ids
-> get_batch 得到 x, y
-> transformer_lm(x)
-> logits
-> cross_entropy(logits, y)
-> backward
-> gradient clipping
-> AdamW step
-> checkpoint
```

## 1. Linear

概念：不带 bias 的线性层，把最后一维从 `d_in` 映射到 `d_out`。

位置：Transformer 里 Q/K/V projection、output projection、FFN、lm_head 都是 linear。

输入输出：

```text
in_features: (..., d_in)
weights: (d_out, d_in)
output: (..., d_out)
```

公式：

```text
output = in_features @ weights.T
```

易错点：权重是 `(d_out, d_in)`，所以要转置。

## 2. Embedding

概念：用 token id 查 embedding table。

位置：语言模型第一层，把整数 token id 变成向量。

输入输出：

```text
weights: (vocab_size, d_model)
token_ids: (...)
output: (..., d_model)
```

实现思路：

```text
weights[token_ids]
```

## 3. SiLU

概念：激活函数。

公式：

```text
SiLU(x) = x * sigmoid(x)
sigmoid(x) = 1 / (1 + exp(-x))
```

位置：SwiGLU FFN 里用作门控激活。

输入输出：shape 不变。

## 4. Softmax

概念：把 logits 转成概率分布。

公式：

```text
softmax(x_i) = exp(x_i) / sum_j exp(x_j)
```

稳定实现：

```text
x = x - max(x, dim, keepdim=True)
exp_x = exp(x)
output = exp_x / sum(exp_x, dim, keepdim=True)
```

易错点：必须 `keepdim=True`，否则广播 shape 容易错。

## 5. Cross Entropy

概念：衡量模型预测和真实 target 的差距。

输入输出：

```text
inputs: (batch_size, vocab_size)
targets: (batch_size,)
loss: scalar
```

稳定公式：

```text
loss_i = log(sum(exp(stable_logits))) - stable_logits[target_i]
loss = mean(loss_i)
```

易错点：

```text
row_indices 要放在 inputs.device 上
```

## 6. RMSNorm

概念：对每个 token 的最后一维向量做 RMS 归一化。

位置：CS336 使用 pre-norm Transformer block，所以 attention 和 FFN 前都会 RMSNorm。

输入输出：

```text
in_features: (..., d_model)
weights: (d_model,)
output: (..., d_model)
```

公式：

```text
rms = sqrt(mean(x^2, dim=-1, keepdim=True) + eps)
output = x / rms * weights
```

## 7. SwiGLU

概念：FFN 的门控版本。一路生成 gate，一路生成 value，然后逐元素相乘。

位置：Transformer block 的 FFN 子层。

输入输出：

```text
in_features: (..., d_model)
w1_weight: (d_ff, d_model)
w3_weight: (d_ff, d_model)
w2_weight: (d_model, d_ff)
output: (..., d_model)
```

公式：

```text
hidden = SiLU(W1 x) * (W3 x)
output = W2 hidden
```

shape：

```text
(..., d_model)
-> W1/W3
-> (..., d_ff)
-> elementwise multiply
-> (..., d_ff)
-> W2
-> (..., d_model)
```

## 8. Scaled Dot-Product Attention

概念：Q 和 K 算相关性，softmax 得注意力权重，再加权 V。

位置：Transformer attention 的核心。

输入输出：

```text
Q: (..., queries, d_k)
K: (..., keys, d_k)
V: (..., keys, d_v)
mask: (..., queries, keys)
output: (..., queries, d_v)
```

公式：

```text
scores = Q @ K.transpose(-2, -1)
scores = scores / sqrt(d_k)
scores[mask == False] = very negative number
weights = softmax(scores, dim=-1)
output = weights @ V
```

易错点：

```text
是除以 sqrt(d_k)，不是 scores ** 0.5
mask=True 表示可以 attend
```

## 9. Multi-Head Self-Attention

概念：同一个输入 x 生成 Q/K/V，拆成多个 head，每个 head 独立 attention，再拼回。

位置：Transformer block 的 attention 子层。

shape：

```text
x: (..., seq, d_model)
Q/K/V projection: (..., seq, d_model)
split heads: (..., num_heads, seq, d_head)
attention output: (..., num_heads, seq, d_head)
concat heads: (..., seq, d_model)
output projection: (..., seq, d_model)
```

其中：

```text
d_head = d_model // num_heads
```

causal mask：

```text
torch.tril(torch.ones(seq, seq)).bool()
```

## 10. RoPE

概念：Rotary Position Embedding，旋转位置编码。它把向量每两维一组，根据 token 位置旋转。

位置：作用在 Q 和 K 上，不作用在 V 上。

输入输出：

```text
in_query_or_key: (..., sequence_length, d_k)
token_positions: (..., sequence_length)
output: (..., sequence_length, d_k)
```

频率：

```text
freq_i = 1 / theta^(2i / d_k)
i = 0, 1, ..., d_k/2 - 1
```

角度：

```text
angles = token_positions[..., None] * freqs
```

旋转：

```text
x_even = x[..., 0::2]
x_odd = x[..., 1::2]

rotated_even = x_even * cos - x_odd * sin
rotated_odd = x_even * sin + x_odd * cos
```

交错放回：

```text
out[..., 0::2] = rotated_even
out[..., 1::2] = rotated_odd
```

易错点：

```text
theta 不是角度，angles 才是传给 cos/sin 的角度
freqs 和 positions 要和 x 同 device、同 dtype
out 不要写 out = x，避免原地改输入
```

## 11. MHA with RoPE

概念：普通 MHA 加一步，对 split heads 后的 Q/K 做 RoPE。

流程：

```text
Q/K/V projection
-> split heads
-> Q = rope(Q)
-> K = rope(K)
-> V 不变
-> causal attention
-> concat
-> output projection
```

注意 RoPE 的 `d_k` 是：

```text
d_head = d_model // num_heads
```

token positions 广播：

```text
Q shape: (batch, head, seq, d_head)
token_positions: (batch, seq)
token_positions_for_rope = token_positions.unsqueeze(-2)
shape: (batch, 1, seq)
```

## 12. Transformer Block

概念：一个 Transformer 层，由 attention 子层和 FFN 子层组成。

CS336 使用 pre-norm：

```text
x = x + Attention(RMSNorm(x))
x = x + FFN(RMSNorm(x))
```

输入输出：

```text
in_features: (batch, seq, d_model)
output: (batch, seq, d_model)
```

权重字典 key：

```text
attn.q_proj.weight
attn.k_proj.weight
attn.v_proj.weight
attn.output_proj.weight
ln1.weight
ln2.weight
ffn.w1.weight
ffn.w2.weight
ffn.w3.weight
```

## 13. Transformer LM

概念：完整语言模型，输入 token ids，输出每个位置对 vocab 的 logits。

流程：

```text
in_indices
-> embedding
-> Transformer Block 0
-> ...
-> Transformer Block num_layers - 1
-> final RMSNorm
-> lm_head linear
-> logits
```

shape：

```text
in_indices: (batch, seq)
embedding: (batch, seq, d_model)
blocks: (batch, seq, d_model)
lm_head: (batch, seq, vocab_size)
```

层权重处理：

```text
完整 key: layers.0.attn.q_proj.weight
block key: attn.q_proj.weight
```

做法：

```text
每层取 prefix = f"layers.{layer_idx}."
把 prefix 去掉，组成 block_weights
```

## 14. get_batch

概念：从一维 token dataset 里随机切训练样本。

输入输出：

```text
dataset: (num_tokens,)
x: (batch_size, context_length)
y: (batch_size, context_length)
```

关系：

```text
y 是 x 向右错一位
```

向量化索引：

```text
starts: (batch_size,)
offsets: (context_length,)
x_indices = starts[:, None] + offsets[None, :]
x = dataset[x_indices]
y = dataset[x_indices + 1]
```

合法起点：

```text
0 <= start < len(dataset) - context_length
```

## 15. Gradient Clipping

概念：如果所有参数梯度的整体 L2 norm 太大，就整体缩小。

位置：

```text
loss.backward()
gradient_clipping(parameters, max_l2_norm)
optimizer.step()
```

公式：

```text
total_norm = sqrt(sum(all grad^2))
if total_norm > max_l2_norm:
    scale = max_l2_norm / total_norm
    grad *= scale
```

易错点：

```text
跳过 grad is None
只改 parameter.grad，不改 parameter.data
```

## 16. AdamW

概念：optimizer，维护梯度的一阶矩和二阶矩，用来更新参数。

state：

```text
step
exp_avg = m
exp_avg_sq = v
```

公式：

```text
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad^2
m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)
update = m_hat / (sqrt(v_hat) + eps)
```

AdamW decoupled weight decay：

```text
p = p * (1 - lr * weight_decay)
p = p - lr * update
```

PyTorch Optimizer 结构：

```text
defaults 存 lr、betas、eps、weight_decay
self.param_groups 是参数组列表
self.state[p] 存每个参数自己的历史状态
```

易错点：

```text
参数更新要放在 torch.no_grad()
bias correction 不要用 div_ 改坏 exp_avg
addcmul_ 用 value，不是 alpha
```

## 17. Cosine LR Schedule

概念：根据 iteration 返回当前学习率。

三段：

```text
1. warmup: 从 0 线性升到 max_lr
2. cosine decay: 从 max_lr 降到 min_lr
3. after: 保持 min_lr
```

公式：

```text
if it < warmup_iters:
    lr = max_lr * it / warmup_iters
elif it <= cosine_cycle_iters:
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    lr = min_lr + 0.5 * (1 + cos(pi * progress)) * (max_lr - min_lr)
else:
    lr = min_lr
```

## 18. Checkpoint

概念：训练存档。

保存：

```text
model.state_dict()
optimizer.state_dict()
iteration
```

为什么保存 optimizer：AdamW 里有 `exp_avg`、`exp_avg_sq`、`step`，不保存会让 optimizer 失忆。

实现：

```text
torch.save(checkpoint, out)
checkpoint = torch.load(src)
model.load_state_dict(...)
optimizer.load_state_dict(...)
return iteration
```

## 19. Tokenizer Decode

概念：token ids 转回字符串。

流程：

```text
ids
-> vocab[id] 得到 bytes
-> b"".join(...)
-> decode("utf-8", errors="replace")
```

## 20. Tokenizer Encode

概念：字符串转 token ids。

GPT-2 风格流程：

```text
text
-> 先按 special tokens 切开
-> 普通文本做 pre-tokenization
-> 每个 pretoken 做 BPE merge
-> bytes token 查 id
```

special token：

```text
像 <|endoftext|> 必须作为整体 token
不能被 BPE 拆开
```

普通文本 pre-tokenization pattern：

```text
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

注意：这个 pattern 需要 `regex` 库，不是 Python 标准库 `re`。

BPE encode：

```text
初始 tokens = 每个 byte 一个 bytes token
反复找当前 tokens 中 rank 最小的 merge pair
合并该 pair
直到没有可合并 pair
最后查 token_to_id
```

易错点：

```text
普通文本不能整个 part 直接 BPE，必须先 pre-tokenize
否则 "\n\ntesting!" 会错误合成双换行 token
special tokens pattern 要按长度从长到短
普通文本 ids 用 extend，special token id 用 append
```

## 21. encode_iterable

概念：流式编码多段文本，避免一次性把大文件读入内存。

实现思路：

```text
for text in iterable:
    for token_id in encode(text):
        yield token_id
```

## 22. train_bpe

概念：从语料学习 BPE vocab 和 merges。

输入输出：

```text
input_path
vocab_size
special_tokens

return vocab, merges
```

初始 vocab：

```text
0..255 -> 单字节 bytes
special tokens -> special_token.encode("utf-8")
```

训练流程：

```text
1. 读文本
2. special tokens 先切开并跳过
3. GPT-2 pre-tokenization
4. pretoken 转 tuple[bytes]
5. 统计 word_freq
6. 统计 pair_counts
7. 反复选择最高频 pair
8. 合并 pair
9. 记录 merges，新增 vocab
```

tie-break：

```text
best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
```

意思：

```text
先比频率
频率相同，选字典序更大的 pair
```

朴素复杂度：

```text
M = merge 次数
N = 所有 word 的 token 总长度
每轮全量扫描 count_pairs + merge_pair
复杂度约 O(MN)
```

优化思路：

```text
维护 pair_counts
维护 pair_to_words
每次只更新 affected_words
old_word 消失：减掉它贡献的 pairs
new_word 出现：加上它贡献的 pairs
```

最终速度测试通过，说明实现不只是正确，而且足够快。

## 23. Windows 环境问题

Tokenizer 测试里的两个问题：

1. Windows 没有 Unix 的 `resource` 模块。
2. 读取 GPT-2 merges 时需要 UTF-8。

解决：

```powershell
$env:PYTHONUTF8='1'
```

两个 memory usage 测试在 Windows 上 skipped 是正常的：

```text
rlimit support for non-linux systems is spotty
```

这不是代码失败。

## 24. 最终完成情况

你已经通过了主要测试：

```text
test_model.py
test_nn_utils.py
test_data.py
test_optimizer.py
test_serialization.py
test_train_bpe.py
test_tokenizer.py
```

Windows 下 memory usage 测试 skipped 属于环境限制。

从学习成果看，你已经完成：

```text
一个最小 Transformer LM
一套训练工具
一个 GPT-2 风格 BPE tokenizer
一个可通过速度测试的 BPE trainer
```

## 25. 复习建议

最值得反复复习的 shape：

```text
Linear: (..., d_in) -> (..., d_out)
MHA: (..., seq, d_model) -> (..., heads, seq, d_head)
RoPE: (..., seq, d_k) -> (..., seq, d_k)
Transformer LM: (batch, seq) -> (batch, seq, vocab_size)
get_batch: dataset -> x/y both (batch, context_length)
```

最值得反复复习的概念：

```text
Q/K/V 的作用
RoPE 为什么只作用 Q/K
pre-norm residual block
AdamW state 和 bias correction
BPE encode 和 train_bpe 的区别
special token 为什么要在 BPE 前处理
```

