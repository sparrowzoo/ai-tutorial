# Softmax 交叉熵公式总结

# 链式求导法则

```
见到常子转负幂  除法转幂函数

加三乘除指log幂

链式法则要牢记

复合广播加一起 高数下册 76页

提合拼凑求化简 

求导口决破玄机
```
# 公式



> $L = -\frac{1}{N}\sum_{i=1}^{N} y_i \log\left(\frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}\right)$，其中 $softmax=p_i = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}$，$L = -\frac{1}{N}\sum_{i=1}^{K} y_i \log(p_i),x_i=logits=XW^T+B$
> 
> 



$\boxed{
\begin{aligned}
\text{Softmax:} & \quad p_i = \frac{e^{x_i}}{\sum_j e^{x_j}} \\[10pt]
\text{Cross Entropy:} & \quad L = -\frac{1}{N}\sum_i y_i \log(p_i) \\[10pt]
\text{Softmax + CE:} & \quad L = -\frac{1}{N}\sum_i y_i \log\left(\frac{e^{x_i}}{\sum_j e^{x_j}}\right)
\end{aligned}
}$

# 计算图

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NjE1ZGQ0MTQxMDQ2ODUwZjU3YzNhMTc5ZmVkOTYyMzdfYzNlZTE5N2I5ZmM1ZjI1NzM0MTdmNDY3ZjE1OTMwYmZfSUQ6NzY2MTg0NDc3ODA1MjcwMTQyMV8xNzg0ODA2NjQzOjE3ODQ4OTMwNDNfVjM)

化简

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=M2FmZDZiNmI4YWVjOGQ1Yjk3OTE1NjQ4MGViNDQxYWRfNTAzYmFiYmU4NzAwMTEyYzRjZWE2OWUwNTk4ODE2NDNfSUQ6NzY2MjU3MjgxMjMzMDYxNzc4Nl8xNzg0ODA2NjQzOjE3ODQ4OTMwNDNfVjM)

# 拆解公式

- 见到负号乘\-1

$L = -1*Z$，其中 $Z = \sum_{i=1}^{K} y_i \log\left(\frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}\right)$, $L_z'=-1$

- 加法

$Z = \sum_{i=1}^{K} Z_i$ 其中 $Z_i =y_i \log\left(\frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}\right)$,$Z'_{z_i}=1,Z_{z_i}=-1,$

- 乘法

$Z_i =y_i *L_i$ ,其中 $L_i = log\frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}$,$Z{_i}'_{L_i}=y_i,Z'_{L_i}=-y_i,$

- log 

$L_i = log P_i$ ,其中$P_i =\frac{e^{x_i}}{S}$ ,$L{_i}'_{P_i}=\frac{S}{e^{x_i}}$ ，其中 $S=\sum_{j=1}^{K} e^{x_j} $，参考$ln'(x)=\frac{1}{x}$,$Z'_{P_i}=\frac{-Sy_i}{e^{x_i}},$

- 乘法

$P_i = {E_i}*S_0$  ，其中$S_0=\frac{1}{S}=S^{-1}$$,S=\sum_{j=1}^{K} E_j,E_i=e^{x_i},$ $P{_i}'_{E_i}=\frac{1}{S},  Z'_{E_i}=\frac{-y_i} {e^{x_i}};P{_i}'{_S{_0}}=E_{i}=e^{x_i},Z'_{S_{0}}=-Sy_i;$ 公式1$Z'_{E_i}=\frac{-y_i} {e^{x_i}};$

- 见到常子转负幂

$S_0=\frac{1}{S}=S^{-1}$ $S_0'{_{S}}=-S{^{-2}}$,所以$Z'_S=-S*\sum_{i}y_i * -S^{-2}=\sum_{i}y_i*S^{-1}=\frac{\sum_{i}y_i}{S},因为y_i是独热=\frac{1}{S}$

- 加法

$S=\sum_{j=1}^{K} E_i$,其中$E_i=e^{x_i}$ $S'_{E_i}=1$ $Z'_{E_i}=\frac{1}{S}$ 公式2

- 指数

$E_i=e^{x_i}$ $E'_x=e^{x_i}$ ,



由公式1，2得 $Z'x_i=e^{x_i}(\frac{1}{S}+\frac{-y_i} {e^{x_i}})=\frac{e^{x_i}}{S}-y_i=\hat{y_i}-y_i$





# 如何求损失和梯度？

## 损失

对于 N 个样本、C 个类别 的分类问题：

- `target` 是 one\-hot 编码的真实标签，形状为 `[N, C]`

- `proba` 是 softmax 输出的预测概率，形状为 `[N, C]`

交叉熵损失的定义是：

$\mathcal{L} = -\frac{1}{N} \sum_{n=1}^{N} \sum_{c=1}^{C} y_{n,c} \log(p_{n,c})
$

$L = -\frac{1}{N}\sum_{i=1}^{K} y_i \log\left(\frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}\right)$，其中 $p_i = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}$，$L = -\frac{1}{N}\sum_{i=1}^{K} y_i \log(p_i)$



用矩阵运算表示就是：

$\mathcal{L} = -\frac{1}{N} \text{sum}(\text{target} \odot \log(\text{proba}))
$

$单条损失=−y_{true}⋅log(p_{pred})$

其中 ⊙ 表示逐元素相乘。单样本用@也是可以的



## 梯度如何求？

$logits=Y=WX+B$

这里的*X*是形状为N×m的矩阵，m就是输入神经元的个数；而*W*是形状为m×n的权重矩阵，n就是输出神经元的个数。

根据矩阵求导的运算法则，可以得到损失函数*L*关于*X*、*W*的偏导数：

$
 \frac{\partial \mathcal{L}}{\partial X} = \frac{\partial \mathcal{L}}{\partial Y} \cdot W^T
$



$\frac{\partial \mathcal{L}}{\partial W} = X^T \cdot \frac{\partial \mathcal{L}}{\partial Y}
$



![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=Y2Q2MjQwNWZhM2I2NzY2ZWFiMTYyOWQyY2U2ZDllZDFfZGNlMzQzZWFkYWUzODkzMmQ0MDVhMmNkOTYwNzFlMzhfSUQ6NzY2MjU3MTcwODkzOTExMTcyM18xNzg0ODA2NjQzOjE3ODQ4OTMwNDNfVjM)



### 数学推导



根据链式法则，损失 $\mathcal{L}$ 对权重 $W$ 的梯度为：



$\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial \text{logits}} \cdot \frac{\partial \text{logits}}{\partial W}$  其中$logits=Y=WX+B$



**第一项**：对于 Softmax \+ 交叉熵损失，有：



$\frac{\partial \mathcal{L}}{\partial \text{logits}} = \text{proba} - \text{target}$



**第二项**：因为 $\text{logits} = X \cdot W$，所以：



$\frac{\partial \text{logits}}{\partial W} = X^T$



将两项相乘，得到权重的梯度：

$\text{grad}_W = X^T \cdot (\text{proba} - \text{target})$

