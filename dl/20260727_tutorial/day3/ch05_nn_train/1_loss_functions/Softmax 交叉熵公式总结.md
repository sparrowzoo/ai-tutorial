# Softmax 交叉熵公式总结

# 链式求导法则
```
见到负号乘负一，
见到常子转负幂，
加乘除幂看公式，
三指 log 是小弟，
复合函数层层积
多路广播加一起
```

# 公式



> $L = -\sum_{i=1}^{K} y_i \log\left(\frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}\right)$，其中 $p_i = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}$，$L = -\sum_{i=1}^{K} y_i \log(p_i)$
> 
> 



$\boxed{
\begin{aligned}
\text{Softmax:} & \quad p_i = \frac{e^{x_i}}{\sum_j e^{x_j}} \\[10pt]
\text{Cross Entropy:} & \quad L = -\sum_i y_i \log(p_i) \\[10pt]
\text{Softmax + CE:} & \quad L = -\sum_i y_i \log\left(\frac{e^{x_i}}{\sum_j e^{x_j}}\right)
\end{aligned}
}$

# 计算图

![softmax](softmax.png)

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

$E_i=e^{x_i}$ $E'_x=e^{x_i}$ ,由公式1，2得 $Z'x_i=e^{x_i}(\frac{1}{S}+\frac{-y_i} {e^{x_i}})=\frac{e^{x_i}}{S}-y_i=\hat{y_i}-y_i$



