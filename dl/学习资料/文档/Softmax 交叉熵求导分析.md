# Softmax 交叉熵求导分析

# 链式求导法则
见到常子转负幂  除法转幂函数
幂指log三加乘除  

链式法则要牢记

复合广播加一起 高数下册 76页

提合拼凑求化简 

求导口决破玄机

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

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZTNiZGM5YjU5Nzg0NGZmNzdhNjk0ZmFmZTlmOWRlZTRfNjE1MTlmNTA5Zjk0MDQ5NmI5NmNmZWRlOTQ0MTNkMzNfSUQ6NzY2MTg0NDc3ODA1MjcwMTQyMV8xNzg0OTcxNDI3OjE3ODUwNTc4MjdfVjM)



化简

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=OTQxYzIyYTBmODI1ODcxNDEwZWQwOWY5ZTUzYjdmZTNfNGU2NjFkNzJjMGVkNjlhM2UxMGE3ZWM5OGI2MjhjZmFfSUQ6NzY2MjU3MjgxMjMzMDYxNzc4Nl8xNzg0OTcxNDI3OjE3ODUwNTc4MjdfVjM)

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





## 同济《高等数学》第七版下册 P76 多元复合函数求导法则

设 $z = f(u, v)$，其中 $u = \varphi(t)$，$y = \psi(t)$，则：

$\boxed{\frac{dz}{dt} = \frac{\partial z}{\partial u} \cdot \frac{du}{dt} + \frac{\partial z}{\partial v} \cdot \frac{dv}{dt}}$



![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YTM4OTdlZGMxODNhNmEzZDM2ODgxMjYwZTNkYzFiZTlfYTZmNmYyMzU1ZjRlNzdmOWQ1ZjdlZGE4YWVlY2NlNTZfSUQ6NzY2NjM2NDI2MzE1MzQ1NDA1OV8xNzg0OTcxNDI3OjE3ODUwNTc4MjdfVjM)



![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=OTg2MjYxZTQ1ZTVlZjU3Yjc2NzMzYTczMTdmNWE5NGRfZDI2ZWZmN2QzZjhhYTUzNTliY2IxYmQ5M2Y4YTFmZWRfSUQ6NzY2NjM2NDI2MTI4MzAwNzc4OV8xNzg0OTcxNDI3OjE3ODUwNTc4MjdfVjM)



---



若 $z = f(u, v)$，其中 $u = \varphi(x, y)$，$v = \psi(x, y)$，则：

$\boxed{\frac{\partial z}{\partial x} = \frac{\partial z}{\partial u} \cdot \frac{\partial u}{\partial x} + \frac{\partial z}{\partial v} \cdot \frac{\partial v}{\partial x}}$

$\boxed{\frac{\partial z}{\partial y} = \frac{\partial z}{\partial u} \cdot \frac{\partial x}{\partial y} + \frac{\partial z}{\partial v} \cdot \frac{\partial v}{\partial y}}$







