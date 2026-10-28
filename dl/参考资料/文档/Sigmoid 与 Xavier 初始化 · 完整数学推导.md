# Sigmoid 与 Xavier 初始化 · 完整数学推导

# **目标：让信号在每一层稳定传播**

## 方差一致性要求：Var\(z\) = Var\(X\)

输入标准化后 Var\(X\) = 1，所以目标为 **Var\(z\) = 1**

## **线性层输出方差公式**



线性层 $z = W \cdot X + b$

设 $z = w_1 x_1 + w_2 x_2 + \dots + w_n x_n$，假设独立且均值为 0：

$\text{Var}(z) = \sum_i^n \text{Var}(w_i x_i) = \sum_i^n \text{Var}(w_i) \cdot \text{Var}(x_i)$

如果所有 $w_i$ 同分布，所有 $x_i$ 同分布，则：

$\text{Var}(z) = n \cdot \text{Var}(W) \cdot \text{Var}(X)$ 其中 **n = 输入维度**（上一层神经元数量）

## **第 3 步：输入标准化 → 控制 Var\(X\)**

**标准化方法：****StandardScaler**

为什么是 1？特征一般会标准化 $x′=\frac{x−μ}{σ}$,经过变换后，新的数据集 *x*′ 将具有以下性质：均值 = 0,方差 = 1。

使得 E\[X\] = 0，Var\(X\) = 1

这是所有初始化方法的前提条件。

## **第 4 步：方差一致性 → 反推 Var\(W\)**

代入目标值：

方差一致性目标：**Var\(z\) = 1**

代入$\text{Var}(z) = n \cdot \text{Var}(W) \cdot \text{Var}(X)=1,已知Var(X) = 1$ ，：

**Var\(W\) = 1/n**

> ✅ **结论：** 权重的方差必须与输入维度 n 成反比。这是 Xavier 初始化的数学根源。
> 
> 

## **第 5 步：Xavier 初始化的两种形式**

### **正态分布**

- $W \sim \mathcal{N}(0, \sigma^2)
,Var(W)=\sigma^2=\frac{1}{n},σ = \sqrt{\frac{1}{n}}$ 

- **前向传播**：要求 `Var(z) = Var(X)`，推出 `Var(W) = 1 / n_in`

- **反向传播**：要求 `Var(梯度) = Var(输出梯度)`，推出 `Var(W) = 1 / n_out`

为了同时满足前向和反向传播，Xavier 取两者的调和平均：

$\text{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}}=\sigma^2,σ = \sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}}$

### **均匀分布**

均匀分布 $W \sim \mathcal{U}(-a, a)$ 的：

- **均值**：$\mu = 0$

- **方差**：$\text{Var}(W) = \frac{a^2}{3}$

- **标准差**：$\sigma = \frac{a}{\sqrt{3}}$

$a$ 是均匀分布的**边界值**，表示取值范围为 $[-a, a]$。

- $W \sim \mathcal{U}(-a, a),\text{Var}(W) = \frac{a^2}{3} = \frac{1}{n},a = \sqrt{\frac{3}{n}}$

同样考虑前后向传播调和平均后得

$a = \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}$

## **第 6 步：3σ 原则 → Var\(z\)=1 保证 z 落在有效区间**

### **参考引用 3σ 原则** 

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=Yjg5NmRhMGNmYjk1YmUwZDM5MDlhN2Y1ZjU4MGMwOGJfNTdmZDZmYTRmODA4YjkwMGQ2MTljMDRjN2I0ODVhMGZfSUQ6NzY2NTE1ODk1MTU0MTIzMDc5NF8xNzg0Njg1NjQ5OjE3ODQ3NzIwNDlfVjM)

当 Var\(z\) = 1 时，z 的标准差 σ\_z = 1

约 **95%** 的数据落在 \[\-2, 2\] 范围内 约 **99\.7%** 的数据落在 \[\-3, 3\] 范围内

### **Sigmoid 有效区间**

- 有效范围：z ∈ \[\-2, 2\],导数 σ'\(z\) ≥ 0\.1

- **✔ Var\(z\)=1 → 95% 数据落在有效区**

### **Sigmoid 危险区**

- 饱和区：\|z\| \> 5,导数 σ'\(z\) ≈ 0

- **✘ \|z\|\>5 是 6σ 以外的事件，概率极小**

## **Xavier vs Kaiming 对比**

|对比项|Xavier 初始化|Kaiming 初始化|
|---|---|---|
|适用激活函数|Sigmoid / Tanh|ReLU / Leaky ReLU|
|设计目标|Var\(z\) = 1|Var\(ReLU\(z\)\) = 1|
|补偿需求|无|补偿 ReLU 砍掉一半信号|
|权重方差|$\sigma^2=\text{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}}$|$\sigma^2=\text{Var}(W) = \frac{2}{n_{\text{in}}}$|
|正态标准差|$σ = \sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}}$|$σ = \sqrt{\frac{2}{n_{\text{in}}}}$|
|均匀分布边界|$a = \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}$|$a = \sqrt{\frac{6}{n_{\text{in}}}}$|
|PyTorch 方法|xavier\_|kaiming\_|

Kaiming 初始化只考虑输入维度 $n_{\text{in}}$，原因如下：

1. **ReLU 的“砍一半”效应只在前向传播中体现**：

    - 前向传播时，负值被 ReLU 置为 0，有效信号减少一半，方差变为 $\frac{1}{2} \text{Var}(z)$

    - 为了让 ReLU 输出方差仍保持为 1，输入方差需要放大为 2 倍

    - 因此需要 $\text{Var}(W) = \frac{2}{n_{\text{in}}}$

2. **Xavier 考虑 **$n_{\text{in}} + n_{\text{out}}$** 是因为它同时兼顾前向和反向传播**：

    - 前向要求 $\frac{1}{n_{\text{in}}}$

    - 反向要求 $\frac{1}{n_{\text{out}}}$

    - 调和平均得 $\frac{2}{n_{\text{in}} + n_{\text{out}}}$

        

3. **Kaiming 只取 **$n_{\text{in}}$：

    - 因为反向传播在 ReLU 中也会遇到“砍一半”的问题

    - 但反向传播的梯度是从输出层传回的，实际实验中，使用 $n_{\text{in}}$ 就已足够稳定

    - Kaiming He 的论文中给出的标准公式就是 $\text{Var}(W) = \frac{2}{n_{\text{in}}}$

4. **如果非要同时兼顾 **$n_{\text{in}}$** 和 **$n_{\text{out}}$，可以用：

$\text{Var}(W) = \frac{4}{n_{\text{in}} + n_{\text{out}}}$

但 Kaiming 论文中直接使用 $\frac{2}{n_{\text{in}}}$，因为前向传播的主导影响更大。



## 标准化本质



$x' = \frac{x - \mu}{\sigma}$Xavier / Kaiming 本质就是标准化

- 使数据 **均值为 0、方差为 1**

---

## 实际数据范围

**不保证在 \[\-1, 1\] 之间。**

标准化只保证均值为 0、方差为 1，具体数值范围取决于原始数据的分布。

若原始数据服从**正态分布**：

- 约 68% 落在 \[\-1, 1\]

- 约 95% 落在 \[\-2, 2\]

- 约 99\.7% 落在 \[\-3, 3\]

若有**异常值或长尾分布**，数据范围可能远超 \[\-1, 1\]。

## **最终总结**

> # **Xavier 初始化 = 输入标准化 \+ 方差一致性 \+ 3σ 原则**
> 
> 

1. **输入标准化** 保证 Var\(X\) = 1，消除数据尺度差异。

2. **方差一致性** 要求 Var\(z\) = 1，保证信号稳定传播。

3. **方差公式** Var\(z\) = n · Var\(W\) · Var\(X\) 倒推出 **Var\(W\) = 1/n**。

4. **Xavier 初始化** 正是基于这个数学推导设计出来的。

5. **3σ 原则** 保证 Var\(z\)=1 时，z 约 95% 落在 Sigmoid 有效区间 \[\-2, 2\] 内。

6. **结果：Sigmoid 梯度有效，避免梯度消失。**

## **参考文献**

# **原始论文**

- Glorot \& Bengio, [AISTATS 2010](https://proceedings.mlr.press/v9/glorot10a.html) \(Xavier\)

- He et al\., [ICCV 2015](https://arxiv.org/abs/1502.01852) \(Kaiming\)

