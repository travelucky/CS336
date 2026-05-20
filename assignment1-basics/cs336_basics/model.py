import math
import torch
from einops import rearrange
import numpy as np
import regex as re
def linear(
    in_features,weights
) :
    return in_features @ weights.T


def embedding(weights,token_ids):
    return weights[token_ids]


def silu(in_features):
    sigmoid = 1 / (1 + torch.exp(-in_features))
    return in_features * sigmoid

def softmax(in_features,dim):
    # Max = torch.max(in_features[dim])
    tmp_in_features= in_features - torch.max(in_features,dim,keepdim=True).values
    sum_features = torch.sum(torch.exp(tmp_in_features),dim,keepdim=True)
    return torch.exp(tmp_in_features) / sum_features



def cross_entropy(inputs, targets):
    batch_size = inputs.shape[0]
    max_values = torch.max(inputs,dim = -1,keepdim=True).values
    stable_inputs = inputs - max_values
    exp_values = torch.exp(stable_inputs)
    sum_values = torch.sum(exp_values,dim = -1)
    log_sum_values = torch.log(sum_values)
    row_indices = torch.arange(0,batch_size,1,device=inputs.device)
    correct_logits = stable_inputs[row_indices,targets]
    loss_per_sample = log_sum_values - correct_logits
    return loss_per_sample.mean()


def rmsnorm(in_features,weights,eps):
    output = in_features / torch.sqrt(torch.mean(in_features ** 2,dim = -1,keepdim = True) + eps) * weights
    return output


def swiglu(in_features,w1_weight,w2_weight,w3_weight):
    gate_input = linear(in_features,w1_weight)
    gate = silu(gate_input)
    value = linear(in_features,w3_weight)

    hidden = gate * value
    output = linear(hidden,w2_weight)
    return output



def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2,-1)
    scores = scores / (d_k ** 0.5)
    if mask is not None:
        scores = torch.where(mask,scores,float(-1e9))
    weights = softmax(scores,dim = -1)
    return weights @ V


def multihead_self_attention(d_model,num_heads,q_proj_weight,k_proj_weight,v_proj_weight,o_proj_weight,in_features):
    d_head = d_model // num_heads
    x = in_features

    seq_len = in_features.shape[-2]
    Q = linear(x,q_proj_weight)
    K = linear(x,k_proj_weight)
    V = linear(x,v_proj_weight)


    Q = rearrange(Q,'... s (h d) -> ... h s d',h = num_heads)
    K = rearrange(K,'... s (h d) -> ... h s d',h = num_heads)
    V = rearrange(V,'... s (h d) -> ... h s d',h = num_heads)

    mask = torch.tril(torch.ones(seq_len,seq_len,device = in_features.device)).bool()

    attn_out = scaled_dot_product_attention(Q, K, V, mask=mask)

    concat_out = rearrange(attn_out, '... h s d -> ... s (h d)')
    output = linear(concat_out,o_proj_weight)
    return output

def rope(d_k, theta, max_seq_len, in_query_or_key, token_positions):
    x = in_query_or_key

    half_dim = d_k // 2
    i = torch.arange(half_dim, device=x.device, dtype=x.dtype)
    freqs =  1 / theta ** (2 * i / d_k)

    positions = token_positions.to(device=x.device, dtype=x.dtype)[..., None]

    angles = positions * freqs
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd  = x_even * sin + x_odd * cos

    out = torch.empty_like(x)
    out[..., 0::2] = rotated_even
    out[..., 1::2] = rotated_odd
    return out

def multihead_self_attention_with_rope(d_model,num_heads,max_seq_len,theta,q_proj_weight,k_proj_weight,
                                       v_proj_weight,o_proj_weight,in_features,token_positions):
    d_head = d_model // num_heads
    x = in_features

    seq_len = in_features.shape[-2]
    Q = linear(x,q_proj_weight)
    K = linear(x,k_proj_weight)
    V = linear(x,v_proj_weight)

    if token_positions is None:
        token_positions = torch.arange(seq_len,device = x.device)
    Q = rearrange(Q,'... s (h d) -> ... h s d',h = num_heads)
    K = rearrange(K,'... s (h d) -> ... h s d',h = num_heads)
    V = rearrange(V,'... s (h d) -> ... h s d',h = num_heads)


    token_positions_for_rope = token_positions.unsqueeze(-2)

    Q = rope(d_head,theta,max_seq_len, Q, token_positions_for_rope)
    K = rope(d_head,theta,max_seq_len, K, token_positions_for_rope)
    mask = torch.tril(torch.ones(seq_len,seq_len,device = in_features.device)).bool()

    attn_out = scaled_dot_product_attention(Q, K, V, mask=mask)

    concat_out = rearrange(attn_out, '... h s d -> ... s (h d)')
    output = linear(concat_out,o_proj_weight)
    return output

def transformer_block(d_model,num_heads,d_ff,max_seq_len,theta,weights,in_features):

    q_proj_weight = weights["attn.q_proj.weight"]
    k_proj_weight = weights["attn.k_proj.weight"]
    v_proj_weight = weights["attn.v_proj.weight"]
    o_proj_weight = weights["attn.output_proj.weight"]

    ln1_weight = weights["ln1.weight"]
    ln2_weight = weights["ln2.weight"]

    w1_weight = weights["ffn.w1.weight"]
    w2_weight = weights["ffn.w2.weight"]
    w3_weight = weights["ffn.w3.weight"]

    x = in_features
    normed_x = rmsnorm(x,ln1_weight,eps = 1e-5)
    attn_out = multihead_self_attention_with_rope(d_model,num_heads,max_seq_len,theta,q_proj_weight,k_proj_weight,
                                       v_proj_weight,o_proj_weight,normed_x,None)
    
    x = x + attn_out
    normed_x = rmsnorm(x,ln2_weight,eps = 1e-5)
    ffn_out = swiglu(normed_x,w1_weight,w2_weight,w3_weight)
    x = x + ffn_out
    return x


def transformer_lm(vocab_size,context_length,d_model,num_layers,num_heads,d_ff,rope_theta,weights,in_indices):
    x = embedding(weights["token_embeddings.weight"], in_indices)
    for layer_idx in range(num_layers):
        prefix = f"layers.{layer_idx}."
        block_weights = {}
        for key,value in weights.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                block_weights[new_key] = value
        x = transformer_block(d_model,num_heads,d_ff,context_length,rope_theta,block_weights,x)
    x = rmsnorm(x,weights["ln_final.weight"],eps = 1e-5)
    logits = linear(x,weights["lm_head.weight"])
    return logits

def get_batch(dataset,batch_size,context_length,device):
    num_possible_starting_indices = len(dataset) - context_length
    starts = np.random.randint(0,num_possible_starting_indices,size = batch_size)
    offsets = np.arange(context_length)
    x_indices = starts[:,None] + offsets[None,:]
    x = dataset[x_indices]
    y = dataset[x_indices + 1]
    x = torch.tensor(x, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    return x,y


def gradient_clipping(parameters,max_l2_norm):
    grads = [p.grad for p in parameters if p.grad is not None]

    if len(grads) == 0:
        return None
    
    total_norm_sq = 0
    for g in grads:
        total_norm_sq += torch.sum(g ** 2)
    
    total_norm_sq = torch.sqrt(total_norm_sq)

    if total_norm_sq > max_l2_norm:
        scale = max_l2_norm / total_norm_sq

        for g in grads:
            g.mul_(scale)
    return None


class AdamW(torch.optim.Optimizer):
    def __init__(self,params,lr = 1e-3,betas = (0.9,0.999),eps = 1e-8,weight_decay = 0.01):
        defaults = dict(lr = lr,betas = betas,eps = eps,weight_decay = weight_decay)
        super().__init__(params,defaults)

    def step(self,closure = None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        with torch.no_grad():
            for group in self.param_groups:
                lr = group['lr']
                beta1 = group['betas'][0]
                beta2 = group['betas'][1]
                eps = group['eps']
                weight_decay = group['weight_decay']

                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad

                    state = self.state[p]

                    # 初始化 
                    if len(state) == 0 :
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                    exp_avg,exp_avg_sq = state['exp_avg'],state['exp_avg_sq']
                    state['step'] += 1
                    exp_avg = exp_avg.mul_(beta1).add_(grad,alpha = 1 - beta1)
                    exp_avg_sq = exp_avg_sq.mul_(beta2).addcmul_(grad ,grad,value = 1 - beta2)

                    t = state['step']
                    bias_correction1 = 1 - beta1 ** t
                    bias_correction2 = 1 - beta2 ** t

                    m_hat = exp_avg / bias_correction1
                    v_hat = exp_avg_sq / bias_correction2

                    p.mul_(1 - lr * weight_decay)

                    update = m_hat / (v_hat.sqrt() + eps)
                    p.sub_(lr * update)
        return loss        


def get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    elif it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        lr = min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (max_learning_rate - min_learning_rate)
        return lr
    else:
        return min_learning_rate
    
def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = dict(model_state_dict = model.state_dict(),optimizer_state_dict = optimizer.state_dict(),iteration = iteration)
    torch.save(checkpoint,out)

def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['iteration']

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        self.token_to_id = {token: id for id, token in self.vocab.items()}
        
        self.special_token_to_id = {}
        for token in self.special_tokens:
            token_byte = token.encode("utf-8")
            self.special_token_to_id[token] = self.token_to_id[token_byte]

        self.pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.pre_tokenize_pattern = re.compile(self.pat)

    def decode(self, ids):
        token_bytes = [self.vocab[idx] for idx in ids]
        byte_string = b"".join(token_bytes)
        text = byte_string.decode("utf-8", errors="replace")
        return text
    

    def _encode_ordinary_text(self, text):
        ids = []
        for pretoken in self.pre_tokenize_pattern.findall(text):
            ids.extend(self._encode_chunk(pretoken))
        return ids
    
    def _encode_chunk(self, text):
        raw_bytes = text.encode("utf-8")
        tokens = [bytes([b]) for b in raw_bytes]
        merge_dict = {pair: i for i, pair in enumerate(self.merges)}

        while len(tokens) >= 2:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            best_pair = None
            min_value = float('inf')

            for pair in pairs:
                if pair in merge_dict and merge_dict[pair] < min_value:
                    best_pair = pair
                    min_value = merge_dict[pair]

            if best_pair is None:
                break
                
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i+1] == best_pair[1]:
                    new_tokens.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            
        return [self.token_to_id[token] for token in tokens]
    
    def encode(self, text):
        if not self.special_tokens:
            return self._encode_chunk(text)
            
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "(" + "|".join(re.escape(tok) for tok in sorted_special_tokens) + ")"
        
        parts = re.split(pattern, text)
        final_ids = []
        for part in parts:
            if not part:
                continue
            if part in self.special_token_to_id:
                final_ids.append(self.special_token_to_id[part])
            else:
                final_ids.extend(self._encode_ordinary_text(part))
                
        return final_ids

    def encode_iterable(self, iterable):
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id


def init_vocab(special_tokens): # 初始化
    vocab = {i: bytes([i]) for i in range(256)}

    if special_tokens:
        for i, token in enumerate(special_tokens):
            vocab[256 + i] = token.encode("utf-8")

    return vocab


from collections import defaultdict
# 统计全局的相邻两个词频数
# 朴素做法类似一个 组数 * 长度的时间复杂度
def count_pairs(word_freq):  
    pair_counts = defaultdict(int)
    pair_to_words = defaultdict(set)
    for word, freq in word_freq.items():
        for left,right in zip(word,word[1:]):
            pair = (left,right)
            pair_counts[pair] += freq
            pair_to_words[pair].add(word)
    return pair_counts,pair_to_words

# 全局合并最优解

def merge_pair(word_freq, best_pair,affected_words,pair_counts,pair_to_words): 
    to_add = defaultdict(int)
    affected_words = list(affected_words)
    to_remove = []
    for old_word in affected_words:
        if old_word not in word_freq:
            continue

        has_pair = False
        for left, right in zip(old_word, old_word[1:]):
            if (left, right) == best_pair:
                has_pair = True
                break
        if not has_pair:
            continue

        freq = word_freq[old_word]
        to_remove.append(old_word)

        for left,right in zip(old_word,old_word[1:]):
            pair = (left,right)
            if pair in pair_counts:
                pair_counts[pair] -=freq
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
            if pair in pair_to_words:
                pair_to_words[pair].discard(old_word)
                if not pair_to_words[pair]:
                    del pair_to_words[pair]



        new_word = []
        i = 0
        while i < len(old_word):
            if i < len(old_word) - 1 and old_word[i] == best_pair[0] and old_word[i+1] == best_pair[1]:
                new_word.append(best_pair[0] + best_pair[1])
                i += 2
            else:
                new_word.append(old_word[i])
                i += 1
        to_add[tuple(new_word)] += freq

    for old_word in to_remove:
        del word_freq[old_word]

    for new_word,freq in to_add.items():
        word_freq[new_word] += freq

        for left,right in zip(new_word,new_word[1:]):
            pair = (left,right)
            pair_counts[pair] +=freq
            pair_to_words[pair].add(new_word)

    return word_freq

def build_word_freq(text,pre_tokenize_pattern,special_tokens=None):
    special_tokens = special_tokens or []
    word_freq = defaultdict(int)

    if special_tokens:
        sorted_specials = sorted(special_tokens,key = len,reverse=True)
        pattern = "(" + "|".join(re.escape(tok) for tok in sorted_specials) + ")"
        parts = re.split(pattern, text)
    else:
        parts = [text]
    for part in parts:
        if not part:
            continue
        if part in special_tokens:
            continue

        words = pre_tokenize_pattern.findall(part)
        for word in words:
            raw_bytes = word.encode('utf-8')
            bytes_tuple = tuple(bytes([b]) for b in raw_bytes)
            word_freq[bytes_tuple] += 1

    return word_freq
def train_bpe(input_path, vocab_size, special_tokens, **kwargs):
    vocab = init_vocab(special_tokens)
    merges = []
    with open(input_path,'r',encoding='utf-8') as f:
        text = f.read()
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokenize_pattern = re.compile(PAT)
    word_freq = build_word_freq(text,pre_tokenize_pattern,special_tokens)
    pair_counts, pair_to_words = count_pairs(word_freq)

    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        best_pair = max(pair_counts,key = lambda pair:(pair_counts[pair],pair))
        merges.append(best_pair)

        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        word_freq = merge_pair(word_freq, best_pair, pair_to_words[best_pair],pair_counts,pair_to_words)
    return vocab,merges