# Softmax 交叉熵求导分析

# 公式

一个样本,K个类别



> $L = -\sum_{i=1}^{K} y_i \log\left(\frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}\right)$，其中 $softmax=p_i = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}$，$L = -\sum_{i=1}^{K} y_i \log(p_i),x_i=logit=Wx^T+b$
> 
> 



$\boxed{
\begin{aligned}
\text{Softmax:} & \quad p_i = \frac{e^{x_i}}{\sum_j e^{x_j}} \\[10pt]
\text{Cross Entropy:} & \quad L = -\sum_i^K y_i \log(p_i) \\[10pt]
\text{Softmax + CE:} & \quad L = -\sum_i^K y_i \log\left(\frac{e^{x_i}}{\sum_j e^{x_j}}\right)
\end{aligned}
}$



对于 N 个样本、C 个类别 的分类问题：

- `target` 是 one\-hot 编码的真实标签，形状为 `[N, C]`

- `proba` 是 softmax 输出的预测概率，形状为 `[N, C]`

交叉熵损失的定义是：

$\mathcal{L} = -\frac{1}{N} \sum_{n=1}^{N} \sum_{c=1}^{C} y_{n,c} \log(p_{n,c})
$



用矩阵运算表示就是：

$\mathcal{L} = -\frac{1}{N} \text{sum}(\text{target} \odot \log(\text{proba}))
$

$单条损失=−y_{true}⋅log(p_{pred})$

其中 ⊙ 表示逐元素相乘。单样本用@也是可以的

# 计算图

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YTc1YjQ1NzhmODAwYzdmMTFjMGYzMTU5OTY3YTA4NTNfZmJlNGY0MzA4OTgyZWJhMDI2NjRkYjQzNzk0NDU3MDdfSUQ6NzY2MTg0NDc3ODA1MjcwMTQyMV8xNzg2MDA2MzUxOjE3ODYwOTI3NTFfVjM)

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=M2RhYzk0MTczYjE0YmU0ZDg3Njg3YWQ1ZWY1MTA5ODlfMzEyMTlhNWRlOTQyOTAwY2IzOWY3YjUyNGE2N2UzNzVfSUQ6NzY3MDgzMDE2MTY5MjQ0NTkzOF8xNzg2MDA2MzUxOjE3ODYwOTI3NTFfVjM)

化简

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NTM0MTYwNmNmMTdlNjBlMjM5ZmUwOWRiNmI4NTM0MzRfZjA2YTc2OGQ4NjNhOTU0OWY2NDFlMDk5OGFhZWI4ZjBfSUQ6NzY2MjU3MjgxMjMzMDYxNzc4Nl8xNzg2MDA2MzUxOjE3ODYwOTI3NTFfVjM)

# 拆解公式

- 见到负号乘\-1

$L = -1*Z$，其中 $Z = \sum_{i=1}^{K} y_i \log\left(\frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}\right)$,                                                              $L_z'=-1$

- 加法

$Z = \sum_{i=1}^{K} Z_i$ 其中 $Z_i =y_i \log\left(\frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}\right)$,$Z'_{z_i}=1$                                                        $L'_{zi}=-1$

- 乘法

$Z_i =y_i *L_i$ ,其中 $L_i = log\frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}$,$Z{_i}'_{L_i}=y_i$                                                              $L'_{li}=-y$

- log 

$L_i = log P_i$ ,其中$P_i =\frac{e^{x_i}}{S}$ ,$L{_i}'_{P_i}=\frac{S}{e^{x_i}}$ ，其中 $S=\sum_{j=1}^{K} e^{x_j} $，参考$ln'(x)=\frac{1}{x}$        $L'_{pi}=-\frac{y_{i}S}{e^{x_i}}$

- 乘法

$P_i = {E_i}*S_0$  ，其中$S_0=\frac{1}{S}=S^{-1}$$,S=\sum_{j=1}^{K} E_j,E_i=e^{x_i},$ 

$P{_i}'_{E_i}=\frac{1}{S};$                                                  $L'_{Ei}=-\frac{y_{i}}{e^{x_i}} 由E'x=e^{x_i} 等 L'x=-y_i$  公式1

$P{_i}'{_S{_0}}=E_{i}=e^{x_i}$                                       $L'_{SO}=-y_{i}S$

- 见到常子转负幂

$S_0=\frac{1}{S}=S^{-1}$ $S_0'{_{S}}=-S{^{-2}}$,所以$Z'_S=-S*\sum_{i}y_i * -S^{-2}=\sum_{i}y_i*S^{-1}=\frac{\sum_{i}y_i}{S},因为y_i是独热=\frac{1}{S}$\(广播累加\)

- 加法

$S=\sum_{j=1}^{K} E_i$,其中$E_i=e^{x_i}$ $S'_{E_i}=1$  $L'_{Ei}=\frac{1}{S}$

- 指数

$E_i=e^{x_i}$ $E'_x=e^{x_i}$ ,$L'_{x}=\frac{e^{-x}}{S}$公式2



由公式1，2得 $L'x_i=\frac{e^{x_i}}{S}-y_i=\hat{y_i}-y_i$



[多元复合函数求导法则](https://sparrowzoo.feishu.cn/wiki/Aab9wNL68iv1JoktfcQcsgOdnKb)







