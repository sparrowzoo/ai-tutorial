# Softmax 求导

# 公式定义拆解

$y_k = \frac{e^{x_k}}{\sum_{i=1}^{n} e^{x_i}}$

其中 $k = 1, 2, \dots, n$ 设$S = \sum_{i=1}^{n} e^{x_j}$则：$y_k = \frac{e^{x_k}}{S}$
- **分子**：当前这个类别原始得分$x_k$的指数值 $e^{x_k}$。
- **分母**：**所有**类别（从1到n）原始得分的指数值之和 $\sum e^{x_i}$。
- **结果**：当前类别的得分占全体总得分的比例，也就是概率 $y_k$。
需要分两种情况：

1. **当 **$k = i$（对自求导）

2. **当 **$k \neq i$（对别人求导）

# 用除法求导法则

除法求导公式：（上导下不导\-下导上不导）

$\left( \frac{u}{v} \right)' = \frac{u' v - v'u }{v^2}$在这里：$u = e^{x_k}$,$v = S = \sum_{i=1}^{n} e^{x_j}$

---

## 情况 1：$k = i$（对自求导）

当 $k = i$ 时：

$u = e^{x_k}, \quad \frac{\partial u}{\partial x_k} = e^{x_k}$

$v = S, \quad \frac{\partial v}{\partial x_k} = e^{x_k}$,因为其他维作为常数求导，变为0

代入除法法则：

$\frac{\partial y_k}{\partial x_k}
= \frac{e^{x_k} \cdot S - e^{x_k} \cdot e^{x_k}}{S^2}
= \frac{e^{x_k}(S - e^{x_k})}{S^2}$ 拆开后得



$= \frac{e^{x_k}}{S} \cdot \frac{S - e^{x_k}}{S}$ 

根据 Softmax 的定义：$S = \sum_{j=1}^{n} e^{x_j}$，

左边: $\frac{e^{x_k}}{S} =y_k$

右边：$ \frac{S - e^{x_k}}{S} =\frac{S}{S} - \frac{e^{x_k}}{S} = 1 - y_k$

**结论**：$\boxed{\frac{\partial y_k}{\partial x_k} = y_k (1 - y_k)}$

---



## 情况 2：$k \neq i$（对别人求导）



当 $k \neq i$ 时：

$u = e^{x_k}, \quad \frac{\partial u}{\partial x_i} = 0$因为 $e^{x_k}$ 中不包含 $x_i$（$k \neq i$）。

$v = S, \quad \frac{\partial v}{\partial x_i} = e^{x_i}$



代入除法法则：



$\frac{\partial y_k}{\partial x_i}
= \frac{0 \cdot S - e^{x_k} \cdot e^{x_i}}{S^2}
= -\frac{e^{x_k} e^{x_i}}{S^2}$



$= -\frac{e^{x_k}}{S} \cdot \frac{e^{x_i}}{S}
= -y_k \cdot y_i$

**结论**：

$\boxed{\frac{\partial y_k}{\partial x_i} = -y_k y_i}$



## 汇总成完整公式

$\boxed{
\frac{\partial y_k}{\partial x_i} =
\begin{cases}
y_k (1 - y_k), & k = i \\
-y_k y_i, & k \neq i
\end{cases}
}$



