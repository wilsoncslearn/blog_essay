# Transformer 与 GPT 的完整工作流程

## 本章学习目标

完成本章后，应能够解释以下问题。

1. GPT 类模型为什么可以概括为预测下一个 Token。
2. 文本如何经过分词、嵌入和位置编码变成模型可以处理的矩阵。
3. Q、K、V 分别表示什么，自注意力如何计算上下文相关性。
4. 多头注意力与单头注意力有什么区别。
5. FFN、残差连接和层归一化在 Transformer Block 中承担什么作用。
6. 隐藏状态如何转换为词表上的 Logits 和概率分布。
7. 训练阶段与推理阶段在目标、计算过程和参数更新方面有何不同。
8. 自回归生成为什么必须逐 Token 进行，以及 KV Cache 为什么能够提高推理速度。

## 1. GPT 的基本任务

GPT 属于仅解码器结构的自回归语言模型。给定一个 Token 序列

$$
x_1,x_2,\ldots,x_t
$$

模型估计下一个 Token 的条件概率分布

$$
P(x_{t+1}\mid x_1,x_2,\ldots,x_t)
$$

如果需要生成多个 Token，模型会重复执行以下过程。

1. 根据已有序列预测下一个 Token。
2. 从概率分布中选择或采样一个 Token。
3. 将新 Token 追加到已有序列末尾。
4. 使用扩展后的序列继续预测。

因此，GPT 的文本生成可以写成概率分解

$$
P(x_1,x_2,\ldots,x_n)=\prod_{t=1}^{n}P(x_t\mid x_1,x_2,\ldots,x_{t-1})
$$

这称为自回归分解。它把一个完整序列的概率拆成一系列下一 Token 预测问题。

<div style='margin:24px 0;padding:20px;border:1px solid #c7d2fe;border-radius:16px;background:#f8fafc !important;color:#111827 !important;'>
  <div style='margin-bottom:16px;text-align:center;font-size:18px;font-weight:800;color:#312e81 !important;'>GPT 自回归生成的完整闭环</div>
  <div style='display:flex;flex-wrap:wrap;align-items:center;justify-content:center;gap:8px;'>
    <div style='padding:10px 14px;border:1px solid #a5b4fc;border-radius:10px;background:#eef2ff !important;color:#1e1b4b !important;font-weight:700;'>输入文本</div>
    <div style='font-size:22px;color:#6366f1 !important;'>→</div>
    <div style='padding:10px 14px;border:1px solid #a5b4fc;border-radius:10px;background:#eef2ff !important;color:#1e1b4b !important;font-weight:700;'>Token 序列</div>
    <div style='font-size:22px;color:#6366f1 !important;'>→</div>
    <div style='padding:10px 14px;border:1px solid #818cf8;border-radius:10px;background:#e0e7ff !important;color:#1e1b4b !important;font-weight:700;'>Transformer Blocks</div>
    <div style='font-size:22px;color:#6366f1 !important;'>→</div>
    <div style='padding:10px 14px;border:1px solid #a5b4fc;border-radius:10px;background:#eef2ff !important;color:#1e1b4b !important;font-weight:700;'>Logits</div>
    <div style='font-size:22px;color:#6366f1 !important;'>→</div>
    <div style='padding:10px 14px;border:1px solid #a5b4fc;border-radius:10px;background:#eef2ff !important;color:#1e1b4b !important;font-weight:700;'>概率分布</div>
    <div style='font-size:22px;color:#6366f1 !important;'>→</div>
    <div style='padding:10px 14px;border:1px solid #a5b4fc;border-radius:10px;background:#eef2ff !important;color:#1e1b4b !important;font-weight:700;'>下一个 Token</div>
  </div>
  <div style='margin-top:16px;padding:11px 16px;border-radius:10px;background:#312e81 !important;color:#ffffff !important;text-align:center;font-weight:800;'>将新 Token 追加到序列末尾，再次执行同一流程</div>
</div>

## 2. 从文本到输入矩阵

Transformer 不能直接处理自然语言字符串。文本必须依次经过 Tokenization、Token Embedding 和位置信息注入。

### 2.1 Tokenization

Tokenizer 将字符串切分为词表中的离散单元，并将每个单元映射为整数编号。例如，一个中文词可能对应一个 Token，也可能被拆成多个 Token。英文单词、数字、标点和空格也可能拥有各自的切分方式。

假设词表大小为 $V$，序列长度为 $n$，经过分词后得到

$$
[t_1,t_2,\ldots,t_n],\qquad t_i\in\{0,1,\ldots,V-1\}
$$

Token 编号只是离散索引，不包含可以直接参与神经网络计算的连续语义信息。

### 2.2 Token Embedding

模型维护一个可学习的嵌入矩阵

$$
E\in\mathbb{R}^{V\times d_{model}}
$$

其中 $V$ 是词表大小，$d_{model}$ 是模型隐藏维度。根据 Token 编号从矩阵中取出相应行，可以得到每个 Token 的初始向量。

长度为 $n$ 的序列最终形成输入矩阵

$$
X_{token}\in\mathbb{R}^{n\times d_{model}}
$$

矩阵的每一行对应序列中的一个 Token。

### 2.3 位置信息

纯粹的自注意力计算不天然区分 Token 的先后顺序。因此，模型还必须加入位置信息，否则它难以区分我喜欢你和你喜欢我。

不同模型使用的位置机制并不完全相同，常见方案包括可学习位置嵌入、正弦位置编码和旋转位置编码 RoPE。无论采用哪一种方案，目标都是让注意力计算能够感知相对或绝对位置。

可以将模型的初始输入抽象为

$$
X_0=X_{token}+X_{position}
$$

对于采用 RoPE 的模型，位置信息通常不是直接加在 Token Embedding 上，而是在注意力层中作用于 Q 和 K。这里需要根据具体架构区分。

## 3. 因果自注意力

自注意力允许序列中每个位置根据其他位置的信息更新自身表示。GPT 使用的是因果自注意力，也称掩码自注意力。因果二字表示位置 $t$ 只能读取位置 $1$ 到 $t$，不能提前看到未来 Token。

### 3.1 Q、K、V 的生成

给定某一层的输入矩阵

$$
X\in\mathbb{R}^{n\times d_{model}}
$$

通过三个可学习矩阵进行线性投影

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

其中

$$
W_Q,W_K\in\mathbb{R}^{d_{model}\times d_k},\qquad W_V\in\mathbb{R}^{d_{model}\times d_v}
$$

Q、K、V 可以分别作如下理解。

- Q 表示当前位置希望检索的信息类型。
- K 表示每个位置可供匹配的特征。
- V 表示匹配成功后真正用于汇总的内容。

需要注意，Q、K、V 都是由同一个输入矩阵经过不同线性变换得到的。它们不是人为编写的语义标签，而是在训练过程中自动学习形成的表示。

<div style='margin:24px 0;padding:20px;border:1px solid #bae6fd;border-radius:16px;background:#f8fafc !important;color:#111827 !important;'>
  <div style='margin-bottom:16px;text-align:center;font-size:18px;font-weight:800;color:#0c4a6e !important;'>QKV 注意力计算结构</div>
  <div style='display:flex;flex-wrap:wrap;gap:12px;justify-content:center;'>
    <div style='flex:1 1 180px;min-width:160px;padding:14px;border:1px solid #93c5fd;border-radius:12px;background:#eff6ff !important;color:#172554 !important;text-align:center;'>
      <div style='font-size:22px;font-weight:900;color:#1d4ed8 !important;'>Q</div>
      <div style='margin-top:6px;font-weight:700;'>查询向量</div>
      <div style='margin-top:6px;font-size:13px;color:#334155 !important;'>当前位置需要寻找什么信息</div>
    </div>
    <div style='flex:1 1 180px;min-width:160px;padding:14px;border:1px solid #86efac;border-radius:12px;background:#f0fdf4 !important;color:#14532d !important;text-align:center;'>
      <div style='font-size:22px;font-weight:900;color:#15803d !important;'>K</div>
      <div style='margin-top:6px;font-weight:700;'>键向量</div>
      <div style='margin-top:6px;font-size:13px;color:#334155 !important;'>各位置用于接受匹配的特征</div>
    </div>
    <div style='flex:1 1 180px;min-width:160px;padding:14px;border:1px solid #fdba74;border-radius:12px;background:#fff7ed !important;color:#7c2d12 !important;text-align:center;'>
      <div style='font-size:22px;font-weight:900;color:#c2410c !important;'>V</div>
      <div style='margin-top:6px;font-weight:700;'>值向量</div>
      <div style='margin-top:6px;font-size:13px;color:#334155 !important;'>被注意力权重汇总的实际信息</div>
    </div>
  </div>
  <div style='margin:14px 0;text-align:center;font-size:24px;color:#0284c7 !important;'>↓</div>
  <div style='display:flex;flex-wrap:wrap;align-items:center;justify-content:center;gap:9px;'>
    <div style='padding:10px 13px;border-radius:9px;background:#e0f2fe !important;color:#0c4a6e !important;font-weight:700;'>QKᵀ 计算匹配分数</div>
    <div style='font-size:21px;color:#0284c7 !important;'>→</div>
    <div style='padding:10px 13px;border-radius:9px;background:#e0f2fe !important;color:#0c4a6e !important;font-weight:700;'>缩放、掩码与 Softmax</div>
    <div style='font-size:21px;color:#0284c7 !important;'>→</div>
    <div style='padding:10px 13px;border-radius:9px;background:#ffedd5 !important;color:#7c2d12 !important;font-weight:700;'>对 V 加权求和</div>
    <div style='font-size:21px;color:#0284c7 !important;'>→</div>
    <div style='padding:10px 13px;border-radius:9px;background:#dcfce7 !important;color:#14532d !important;font-weight:700;'>上下文化表示</div>
  </div>
</div>

### 3.2 注意力分数

查询矩阵与键矩阵相乘，得到任意两个位置之间的匹配分数

$$
S=QK^T
$$

矩阵 $S\in\mathbb{R}^{n	imes n}$。其中 $S_{ij}$ 表示位置 $i$ 对位置 $j$ 的关注程度。

为了避免维度较大时点积数值过大，标准缩放点积注意力会除以 $\sqrt{d_k}$

$$
\frac{QK^T}{\sqrt{d_k}}
$$

### 3.3 因果掩码

在 GPT 中，当前位置不能访问未来位置。模型使用上三角掩码矩阵 $M$，将所有未来位置的分数设置为负无穷。

$$
M_{ij}=\begin{cases}
0,&j\leq i\\
-\infty,&j>i
\end{cases}
$$

经过 Softmax 后，负无穷对应的概率变为 0，因此位置 $i$ 不会使用位置 $i+1$ 及其之后的信息。

### 3.4 权重归一化与信息汇总

完整的缩放点积注意力为

$$
\operatorname{Attention}(Q,K,V)=\operatorname{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}+M\right)V
$$

Softmax 沿每一行进行归一化，使每个位置对所有可见位置的注意力权重之和为 1。随后使用这些权重对 V 进行加权求和，得到融合上下文后的新表示。

需要纠正一个常见误解。注意力模块并非全部由线性运算构成。Q、K、V 投影以及输出投影属于线性变换，但 Softmax 本身是非线性的。FFN 仍然非常重要，因为它为每个位置提供了更强的逐位置非线性变换能力。

## 4. 多头注意力

单个注意力头只在一个表示子空间中计算相关性。多头注意力使用多组独立参数并行计算，使模型可以在不同子空间中同时学习多种关系。

第 $i$ 个注意力头可写为

$$
head_i=\operatorname{Attention}(XW_Q^{(i)},XW_K^{(i)},XW_V^{(i)})
$$

所有头的输出先进行拼接，再经过输出投影

$$
\operatorname{MHA}(X)=\operatorname{Concat}(head_1,head_2,\ldots,head_h)W_O
$$

其中 $h$ 是注意力头数量，$W_O$ 用于融合不同头的结果并映射回模型隐藏维度。

在分析训练完成的模型时，有时可以观察到某些头更关注指代、局部语法或长距离依赖。但不能将这种现象理解为固定规则。训练前并不存在第几个头必须负责某项任务的人为规定，不同头的功能也可能重叠或发生变化。

<div style='display:flex;flex-wrap:wrap;gap:16px;margin:24px 0;color:#111827 !important;'>
  <div style='flex:1 1 280px;border:1px solid #a5b4fc;border-radius:14px;padding:18px;background:#eef2ff !important;color:#1e1b4b !important;'>
    <div style='font-size:20px;font-weight:800;margin-bottom:10px;color:#4338ca !important;'>多头注意力</div>
    <div style='font-size:15px;line-height:1.8;color:#1e293b !important;'>处理范围　不同 Token 之间</div>
    <div style='font-size:15px;line-height:1.8;color:#1e293b !important;'>核心计算　相关性匹配与信息汇总</div>
    <div style='font-size:15px;line-height:1.8;color:#1e293b !important;'>主要作用　建立上下文依赖关系</div>
  </div>
  <div style='flex:1 1 280px;border:1px solid #6ee7b7;border-radius:14px;padding:18px;background:#ecfdf5 !important;color:#064e3b !important;'>
    <div style='font-size:20px;font-weight:800;margin-bottom:10px;color:#047857 !important;'>FFN</div>
    <div style='font-size:15px;line-height:1.8;color:#1e293b !important;'>处理范围　每个 Token 独立处理</div>
    <div style='font-size:15px;line-height:1.8;color:#1e293b !important;'>核心计算　升维、激活和降维</div>
    <div style='font-size:15px;line-height:1.8;color:#1e293b !important;'>主要作用　增强逐位置非线性表示能力</div>
  </div>
</div>

## 5. 前馈神经网络 FFN

注意力子层完成不同位置之间的信息交互。FFN 则对每个位置的向量独立应用同一组非线性变换。

经典 FFN 形式为

$$
\operatorname{FFN}(x)=W_2\sigma(W_1x+b_1)+b_2
$$

其中 $W_1$ 通常将隐藏维度从 $d_{model}$ 扩展到更高的中间维度 $d_{ff}$，激活函数 $\sigma$ 引入非线性，$W_2$ 再将向量投影回 $d_{model}$。

早期 Transformer 常使用 ReLU。现代大语言模型常见 GELU、SwiGLU 等变体。不同实现的具体公式可能不同，但其共同目标是对每个 Token 的内部特征进行非线性变换。

FFN 对每个位置独立执行，且所有位置共享同一组 FFN 参数。因此，它不会直接让位置之间交换信息。位置之间的交互主要由注意力完成。

## 6. 残差连接与层归一化

### 6.1 残差连接

残差连接将子层输入直接加到子层输出上

$$
y=x+F(x)
$$

它提供了一条不经过复杂变换的信息通路，有利于深层网络中的梯度传播，并减少随着层数增加而出现的优化困难。

### 6.2 LayerNorm

LayerNorm 对单个 Token 向量的特征维度进行归一化。它可以稳定不同层之间的数值分布，并改善训练过程。

现代 GPT 模型通常采用 Pre-Norm 结构。一个简化的 Pre-Norm Transformer Block 可以写为

$$
Y=X+\operatorname{MHA}(\operatorname{LN}(X))
$$

$$
Z=Y+\operatorname{FFN}(\operatorname{LN}(Y))
$$

原始 Transformer 论文使用的典型 Post-Norm 形式则是在残差相加后执行 LayerNorm。具体模型应以其架构定义为准。

<div style='margin:24px 0;padding:20px;border:1px solid #ddd6fe;border-radius:16px;background:#f8fafc !important;color:#111827 !important;'>
  <div style='margin-bottom:16px;text-align:center;font-size:18px;font-weight:800;color:#4c1d95 !important;'>Pre-Norm Transformer Block 简化结构</div>
  <div style='display:flex;flex-wrap:wrap;align-items:center;justify-content:center;gap:8px;'>
    <div style='padding:11px 14px;border:1px solid #c4b5fd;border-radius:10px;background:#f5f3ff !important;color:#2e1065 !important;font-weight:700;'>输入 X</div>
    <div style='font-size:22px;color:#7c3aed !important;'>→</div>
    <div style='padding:11px 14px;border:1px solid #c4b5fd;border-radius:10px;background:#ede9fe !important;color:#2e1065 !important;font-weight:700;'>LayerNorm</div>
    <div style='font-size:22px;color:#7c3aed !important;'>→</div>
    <div style='padding:11px 14px;border:1px solid #c4b5fd;border-radius:10px;background:#ede9fe !important;color:#2e1065 !important;font-weight:700;'>多头注意力</div>
    <div style='font-size:22px;color:#7c3aed !important;'>→</div>
    <div style='padding:11px 14px;border:1px solid #a78bfa;border-radius:10px;background:#ddd6fe !important;color:#2e1065 !important;font-weight:800;'>与 X 残差相加</div>
    <div style='font-size:22px;color:#7c3aed !important;'>→</div>
    <div style='padding:11px 14px;border:1px solid #6ee7b7;border-radius:10px;background:#d1fae5 !important;color:#064e3b !important;font-weight:700;'>LayerNorm 和 FFN</div>
    <div style='font-size:22px;color:#7c3aed !important;'>→</div>
    <div style='padding:11px 14px;border:1px solid #a78bfa;border-radius:10px;background:#ddd6fe !important;color:#2e1065 !important;font-weight:800;'>再次残差相加</div>
    <div style='font-size:22px;color:#7c3aed !important;'>→</div>
    <div style='padding:11px 14px;border:1px solid #c4b5fd;border-radius:10px;background:#f5f3ff !important;color:#2e1065 !important;font-weight:700;'>输出 Z</div>
  </div>
</div>

## 7. 多层堆叠与隐藏状态

完整 GPT 模型由多个 Transformer Block 堆叠而成。第 $l$ 层的输出是第 $l+1$ 层的输入

$$
X^{(l+1)}=\operatorname{Block}^{(l)}(X^{(l)})
$$

随着层数增加，每个位置的隐藏状态不断融合前文信息并经过非线性加工。最终得到

$$
H\in\mathbb{R}^{n\times d_{model}}
$$

其中第 $i$ 行 $h_i$ 表示位置 $i$ 在完整上下文中的最终隐藏状态。

不宜机械地认为浅层只负责语法、深层只负责逻辑。不同层确实可能呈现由局部到抽象的趋势，但模型内部的特征通常是分布式表示，同一种能力可能由多个层和多个模块共同实现。

## 8. 从隐藏状态到词表 Logits

对于下一 Token 预测，模型主要使用当前序列最后一个位置的隐藏状态 $h_t$。通过输出投影矩阵映射到词表维度

$$
z=h_tW_U+b_U
$$

其中

$$
W_U\in\mathbb{R}^{d_{model}\times V},\qquad z\in\mathbb{R}^{V}
$$

向量 $z$ 的每个分量对应词表中一个 Token 的原始分数，这个向量称为 Logits。

许多语言模型采用权重共享，也称 Weight Tying，即输出投影矩阵与输入嵌入矩阵共享参数或互为转置。典型形式为

$$
W_U=E^T
$$

权重共享可以减少参数量，并使输入嵌入空间与输出分类空间保持联系。但并非所有模型都必须采用完全相同的共享方式。

Logits 经过 Softmax 后得到概率分布

$$
P(x_{t+1}=i\mid x_{\leq t})=\frac{\exp(z_i)}{\sum_{j=1}^{V}\exp(z_j)}
$$

概率最大的 Token 是模型当前认为最可能的下一个 Token。

## 9. 推理阶段

推理阶段用于实际生成文本。此时模型参数保持固定，不执行反向传播。

### 9.1 生成步骤

一次生成循环包括以下步骤。

1. 将当前上下文转换为 Token 序列。
2. 执行前向传播，得到最后位置的 Logits。
3. 根据解码策略选择下一个 Token。
4. 将新 Token 追加到上下文。
5. 重复上述过程，直到生成终止符或达到长度限制。

### 9.2 常见解码策略

**贪心解码**每一步都选择概率最高的 Token。结果稳定，但可能缺乏多样性，也可能出现重复。

**Temperature** 调整 Logits 的尺度

$$
P_i=\operatorname{Softmax}\left(\frac{z_i}{T}\right)
$$

当 $T<1$ 时分布更集中，输出更确定。当 $T>1$ 时分布更平坦，输出更多样。

**Top-k** 只保留概率最高的 $k$ 个候选 Token，再从中采样。

**Top-p** 按概率从高到低排序，保留累计概率达到阈值 $p$ 的最小候选集合，再从该集合中采样。

这些策略只改变如何从概率分布中选取 Token，不会在推理时更新模型权重。

### 9.3 KV Cache

自回归生成具有顺序依赖。生成第 $t+1$ 个 Token 时，前面 $t$ 个 Token 的 K 和 V 已经计算过。如果每一步都重新计算整个序列，会产生大量重复工作。

KV Cache 保存历史位置各层注意力中的 K 和 V。生成新 Token 时，只需要计算新位置的 Q、K、V，并让新 Q 与缓存中的历史 K 进行匹配，再汇总历史 V。

KV Cache 显著减少了解码阶段的重复计算，但会占用额外显存。上下文越长、层数越多、并发请求越多，KV Cache 的显存开销通常越大。

## 10. 训练阶段

训练阶段的目标是让模型提高对真实下一个 Token 的预测概率。

### 10.1 Teacher Forcing

训练时，模型可以一次接收完整的真实序列，并通过因果掩码保证每个位置只能看到其左侧内容。这样，一个长度为 $n$ 的序列可以并行产生多个下一 Token 预测结果。

例如输入为

$$
[x_1,x_2,x_3,x_4]
$$

模型分别学习

$$
x_1\rightarrow x_2
$$

$$
[x_1,x_2]\rightarrow x_3
$$

$$
[x_1,x_2,x_3]\rightarrow x_4
$$

训练阶段虽然具有自回归目标，但不需要像推理阶段一样逐 Token 串行完成整个前向计算。因果掩码允许各位置在遵守信息边界的前提下并行训练。

### 10.2 交叉熵损失

假设真实下一个 Token 的编号为 $y$，模型给出的概率为 $P(y)$，单个位置的负对数似然损失为

$$
\mathcal{L}=-\log P(y)
$$

对批次中所有有效位置取平均，可得到训练损失。模型通过反向传播计算梯度，再由优化器更新嵌入矩阵、注意力参数、FFN 参数、LayerNorm 参数和输出层参数。

### 10.3 参数更新

设模型参数为 $	heta$，学习率为 $\eta$，最基础的梯度下降形式为

$$
\theta\leftarrow\theta-\eta\nabla_{\theta}\mathcal{L}
$$

实际大模型训练通常使用 AdamW 等优化器，并结合学习率预热、衰减、梯度裁剪、混合精度和分布式训练等工程技术。

<div style='overflow-x:auto;margin:24px 0;padding:2px;background:#ffffff !important;color:#111827 !important;border-radius:14px;'>
<table style='width:100%;border-collapse:separate;border-spacing:0;border:1px solid #cbd5e1;border-radius:14px;overflow:hidden;font-size:15px;background:#ffffff !important;color:#111827 !important;'>
  <thead>
    <tr style='background:#3730a3 !important;color:#ffffff !important;'>
      <th style='padding:13px;text-align:left;background:#3730a3 !important;color:#ffffff !important;border:none;'>对比维度</th>
      <th style='padding:13px;text-align:left;background:#3730a3 !important;color:#ffffff !important;border:none;'>训练阶段</th>
      <th style='padding:13px;text-align:left;background:#3730a3 !important;color:#ffffff !important;border:none;'>推理阶段</th>
    </tr>
  </thead>
  <tbody style='background:#ffffff !important;color:#111827 !important;'>
    <tr>
      <td style='padding:13px;border-top:1px solid #cbd5e1;background:#eef2ff !important;color:#1e1b4b !important;font-weight:800;'>主要目标</td>
      <td style='padding:13px;border-top:1px solid #cbd5e1;background:#ffffff !important;color:#111827 !important;'>降低下一 Token 预测损失</td>
      <td style='padding:13px;border-top:1px solid #cbd5e1;background:#ffffff !important;color:#111827 !important;'>生成符合当前上下文的 Token</td>
    </tr>
    <tr>
      <td style='padding:13px;border-top:1px solid #cbd5e1;background:#eef2ff !important;color:#1e1b4b !important;font-weight:800;'>真实答案</td>
      <td style='padding:13px;border-top:1px solid #cbd5e1;background:#f8fafc !important;color:#111827 !important;'>存在，用于计算损失</td>
      <td style='padding:13px;border-top:1px solid #cbd5e1;background:#f8fafc !important;color:#111827 !important;'>不存在，需要模型自行选择</td>
    </tr>
    <tr>
      <td style='padding:13px;border-top:1px solid #cbd5e1;background:#eef2ff !important;color:#1e1b4b !important;font-weight:800;'>参数更新</td>
      <td style='padding:13px;border-top:1px solid #cbd5e1;background:#ffffff !important;color:#111827 !important;'>执行反向传播并更新权重</td>
      <td style='padding:13px;border-top:1px solid #cbd5e1;background:#ffffff !important;color:#111827 !important;'>权重固定，只执行前向传播</td>
    </tr>
    <tr>
      <td style='padding:13px;border-top:1px solid #cbd5e1;background:#eef2ff !important;color:#1e1b4b !important;font-weight:800;'>序列计算</td>
      <td style='padding:13px;border-top:1px solid #cbd5e1;background:#f8fafc !important;color:#111827 !important;'>利用因果掩码并行处理多个位置</td>
      <td style='padding:13px;border-top:1px solid #cbd5e1;background:#f8fafc !important;color:#111827 !important;'>生成过程具有逐 Token 顺序依赖</td>
    </tr>
    <tr>
      <td style='padding:13px;border-top:1px solid #cbd5e1;background:#eef2ff !important;color:#1e1b4b !important;font-weight:800;'>典型附加机制</td>
      <td style='padding:13px;border-top:1px solid #cbd5e1;background:#ffffff !important;color:#111827 !important;'>损失函数、优化器和梯度同步</td>
      <td style='padding:13px;border-top:1px solid #cbd5e1;background:#ffffff !important;color:#111827 !important;'>采样策略和 KV Cache</td>
    </tr>
  </tbody>
</table>
</div>

## 11. 完整数据流

下面将 GPT 的完整工作链路按顺序汇总。

### 11.1 输入阶段

文本经过 Tokenizer 转换为 Token 编号。Token 编号经过嵌入矩阵变成连续向量，并结合位置机制形成初始隐藏状态。

### 11.2 Transformer 主干

隐藏状态依次通过多个 Transformer Block。每个 Block 主要包含因果多头自注意力、FFN、残差连接和 LayerNorm。

注意力负责在允许访问的上下文范围内汇总其他位置的信息。FFN 对每个位置进行独立的非线性特征变换。残差连接和归一化负责改善深层网络的训练稳定性。

### 11.3 输出阶段

最后一层隐藏状态经过最终 LayerNorm 和词表输出投影，得到每个位置在整个词表上的 Logits。训练时，所有有效位置的 Logits 可用于计算下一 Token 损失。推理时，主要读取序列最后位置的 Logits，并通过解码策略选择新 Token。

<div style='margin:24px 0;padding:20px;border:1px solid #d1d5db;border-radius:16px;background:#f8fafc !important;color:#111827 !important;'>
  <div style='margin-bottom:16px;text-align:center;font-size:18px;font-weight:800;color:#111827 !important;'>从输入到输出的分层数据流</div>
  <div style='display:flex;flex-direction:column;align-items:stretch;gap:8px;max-width:760px;margin:0 auto;'>
    <div style='padding:12px 16px;border-radius:10px;background:#e0e7ff !important;color:#1e1b4b !important;text-align:center;font-weight:800;'>文本 → Token 编号 → Token Embedding 与位置信息</div>
    <div style='text-align:center;font-size:22px;color:#6b7280 !important;'>↓</div>
    <div style='padding:12px 16px;border-radius:10px;background:#ede9fe !important;color:#2e1065 !important;text-align:center;font-weight:800;'>因果多头自注意力 → 残差与归一化</div>
    <div style='text-align:center;font-size:22px;color:#6b7280 !important;'>↓</div>
    <div style='padding:12px 16px;border-radius:10px;background:#d1fae5 !important;color:#064e3b !important;text-align:center;font-weight:800;'>FFN 逐位置非线性变换 → 残差与归一化</div>
    <div style='text-align:center;font-size:22px;color:#6b7280 !important;'>↓ 重复多个 Block</div>
    <div style='padding:12px 16px;border-radius:10px;background:#fef3c7 !important;color:#78350f !important;text-align:center;font-weight:800;'>最终隐藏状态 → 词表投影 → Logits</div>
    <div style='text-align:center;font-size:22px;color:#6b7280 !important;'>↓</div>
    <div style='padding:12px 16px;border-radius:10px;background:#fee2e2 !important;color:#7f1d1d !important;text-align:center;font-weight:800;'>训练　计算交叉熵并更新参数　｜　推理　采样下一个 Token</div>
  </div>
</div>

## 12. 常见误区

### 误区一　Token 等于单词

Token 是词表中的离散单元，并不必然对应完整单词。一个词可能由多个 Token 构成，一个 Token 也可能包含标点、空格或多个字符。

### 误区二　注意力权重就是模型的完整解释

注意力权重只描述特定层、特定头和特定位置上的信息聚合比例。模型最终输出还受到其他注意力头、FFN、残差通路和多层堆叠的共同影响。因此，不能仅凭单张注意力图完整解释模型的决策原因。

### 误区三　每个注意力头都有固定职责

多头注意力提供多个表示子空间，但每个头的功能由训练自动形成。它们可能存在分工，也可能存在冗余，不能简单地为每个头指定固定的人类可读功能。

### 误区四　FFN 负责 Token 之间通信

FFN 在每个位置上独立应用。Token 之间的信息传递主要发生在注意力子层。FFN 的重点是对已经融合上下文的单个位置表示进行非线性加工。

### 误区五　推理时模型会自动学习聊天内容

常规推理不会更新基础模型参数。对话历史可以作为上下文影响当前输出，但这与通过反向传播修改模型权重不同。

### 误区六　训练必须逐 Token 串行执行

推理生成具有逐 Token 依赖，通常需要串行解码。训练阶段可以利用真实序列和因果掩码并行计算多个位置的预测损失。

### 误区七　模型先想好完整答案再输出

标准自回归模型通常按 Token 逐步生成。某些系统可能额外使用规划、推理轨迹、工具调用或多阶段工作流，但底层语言模型的输出接口仍然建立在逐 Token 概率预测之上。

## 13. 本章总结

Transformer 的核心计算可以概括为两类。

- 自注意力负责不同 Token 之间的上下文信息交互。
- FFN 负责单个 Token 表示内部的非线性变换。

GPT 的完整生成链路为

$$
\text{文本}
\rightarrow
\text{Token}
\rightarrow
\text{嵌入与位置}
\rightarrow
\text{多层 Transformer}
\rightarrow
\text{隐藏状态}
\rightarrow
\text{Logits}
\rightarrow
\text{概率分布}
\rightarrow
\text{下一个 Token}
$$

训练阶段使用真实下一个 Token 计算交叉熵损失，并通过反向传播更新参数。推理阶段固定参数，通过采样策略逐 Token 生成文本，并可使用 KV Cache 避免重复计算历史 K 和 V。

理解这条链路后，QKV、多头注意力、FFN、残差连接、LayerNorm、Logits、Softmax、训练和推理就不再是相互孤立的术语，而是同一套自回归语言建模系统中的不同环节。

## 14. 复习题

1. 为什么 GPT 的自注意力必须使用因果掩码？
2. QKᵀ 的矩阵形状是什么，每个元素表示什么？
3. 为什么注意力分数需要除以 $\sqrt{d_k}$？
4. 多头注意力中的 Concat 与 $W_O$ 分别完成什么工作？
5. FFN 为什么对每个 Token 独立执行，却仍然能够提高模型能力？
6. Pre-Norm 与 Post-Norm 的主要结构差异是什么？
7. Logits 与概率分布之间是什么关系？
8. 为什么训练阶段能够并行处理多个位置，而推理阶段通常需要逐 Token 生成？
9. KV Cache 缓存了什么，为什么能够加速推理？
10. 为什么上下文学习不等于模型参数更新？
