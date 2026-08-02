# Sigmoid 函数详细推导过程 

## 1\. Sigmoid 函数的指数形式

$\sigma(x) = \frac{1}{1 + e^{-x}} = (1 + e^{-x})^{-1}$



高斯核 $e^{-x^2}$

---

## 2\. 用链式法则求导



令 $u = 1 + e^{-x}$，则 $\sigma = u^{-1}$。

### 第一步：外层求导

$\frac{d\sigma}{du} = -u^{-2} = -\frac{1}{(1 + e^{-x})^2}$

### 第二步：内层求导

$\frac{du}{dx} = \frac{d}{dx}(1 + e^{-x}) = 0 + (-1) \cdot e^{-x} = -e^{-x}$



### 第三步：链式法则（相乘）

$\sigma'(x) = \frac{d\sigma}{du} \cdot \frac{du}{dx}
= \left( -\frac{1}{(1 + e^{-x})^2} \right) \cdot \left( -e^{-x} \right)
= \frac{e^{-x}}{(1 + e^{-x})^2}$



---

## 3\. 转化为 $\sigma(x)(1 - \sigma(x))$



### 拆分：

$\sigma'(x) = \frac{e^{-x}}{(1 + e^{-x})^2}
= \frac{1}{1 + e^{-x}} \cdot \frac{e^{-x}}{1 + e^{-x}}$



### 第一项就是：

$\frac{1}{1 + e^{-x}} = \sigma(x)$



### 第二项：

加1再减1后：

$\frac{e^{-x}}{1 + e^{-x}} =  \frac{e^{-x}+1-1}{1 + e^{-x}} = \frac{1+e^{-x}}{1+e^{-x}} - \frac{1}{1 + e^{-x}} = 1 - \frac{1}{1 + e^{-x}} = 1 - \sigma(x)$

### 合并：

$\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$



