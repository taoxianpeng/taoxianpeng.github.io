---
title: "tinyrenderer"
date: 2026-06-06T12:05:47+08:00 
draft: false
---

> tinyrenderer网站：https://haqr.eu/tinyrenderer/

## Bresenham直线绘制方法

### 基本思路

把屏幕的像素当作是一个一个的格子，在屏幕上画线就是给N*M范围内的直线路径上的格子进行填充颜色。

如何确定填充哪些格子呢？

---
首先，先画一个虚拟的线段(斜率为0~1的情况)，如下图所示：

![picture 1](/images/954f9baed532536a379b9051c9a748fd18b585aaa6bed00b2710087957a85208.png)  

先可以第一个填充起始点(x,y)颜色。
![picture 0](/images/b3c21b7ff657eda0f93a2b323011b9829114e8eecfb9868abda06963b7bde248.png)  

然后填充第二个(x+1, y+?)，也就是往x轴方向+1的位置。判断y是否保持还是需要+1。

[TODO]

### 画三角形

根据上面步骤能画出各个方向的直线之后，输入三个点。分别两两相连就能形成一个线框的三角形.

[TODO]

### 加载obj文件

obj文件结构很简单。内容包括了顶点，纹理坐标，法线，组成三角面的数据。

表示顶点的数据以`v`, 纹理坐标以`vt`， 法线坐标以`vn`, 面顶点索引以`f`开头. 都以空格隔开.

```plain
v 0.11526 0.700717 0.0677257
vt  0.167 0.777 0.909
vn  0.347 -0.372 -0.861
f 1429/1493/1429 1426/1490/1426 1425/1489/1425
```

面的数据稍微特殊点，构建模型的最小单位是三角形。而三个顶点组成一个三角形， f的数据就是将顶点，纹理坐标，法线都联系在一起了

```plain
f 1420/1495/1420   1415/1494/1415   1414/3001/1414
   │   │   │         │   │   │        │   │   │
   │   │   └vn[1420] │   │   └vn[1415]│   │   └vn[1414]
   │   └────vt[1495] │   └────vt[1494]│   └────vt[3001]
   └────────v[1420]  └────────v[1415] └────────v[1414]

```

加载模型就是将这些数据每行都读取，然后保存到对应的数组中，以便渲染的时候使用。

### 绘制模型的线框

从上面模型文件中加载的坐标还不能直接绘制到图像上，因为模型文件里的顶点坐标是**模型坐标系下的坐标**。还需要进行一系列的转化：

> 坐标变换可参考： [LearnOpenGL 坐标系统](https://learnopengl-cn.github.io/01%20Getting%20started/08%20Coordinate%20Systems/)

坐标变化流程：
1. 模型转换
```latex
P_world = M_model * P_obj
```
2. 视图变换
```latex
P_view = M_view * P_world
```
3. 投影变换
```latex
P_clip = M_projection * P_view
```
4. 透视除法
```latex
P_ndc = (x_clip / w_clip, y_clip / w_clip, z_clip / w_clip)
```
5. 视口变换
```latex
screen_x = (P_ndc.x * 0.5 + 0.5) * screen_width
screen_y = (0.5 - P_ndc.y * 0.5) * screen_height // Y轴翻转，因为屏幕Y轴正方形向下
```

## 计算 M_view

按照opengl的坐标系，先给一个`[0, 1, 0]`作为向上的方向向量。然后相机位置到目标点位置为前向量。利用前向量和上向量的叉积计算出右向量。由于上向量不一定与前、右向量都是正交的，所以还需要用前向量与右向量进行叉积计算出真正的上向量


```python
import numpy as np

def normalize(v):
    """向量归一化，返回单位向量"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm
```


```python
def lookat(eye: np.array, target: np.array, up: np.array) -> np.array:
    f = normalize(target - eye)
    s = normalize(np.cross(f, up))
    u = normalize(np.cross(s, f))

    T = np.array([
        [1, 0, 0, -eye[0]],
        [0, 1, 0, -eye[1]],
        [0, 0, 1, -eye[2]],
        [0, 0, 0,    1   ],
    ])

    R = np.array([
        [ s[0], s[1], s[2], 0],
        [ u[0], u[1], u[2], 0],
        [-f[0],-f[1],-f[2], 0],
        [    0,    0,    0, 0],
    ])

    V = R @ T
    return V
```


```python
# test
eye = np.array([0, 2, -5])
target = np.array([0, 0, 0])
up = np.array([0, 1, 0])
lookat(eye, target, up)
```




    array([[-1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00],
           [ 0.00000000e+00,  9.28476691e-01,  3.71390676e-01,
            -2.22044605e-16],
           [ 0.00000000e+00,  3.71390676e-01, -9.28476691e-01,
            -5.38516481e+00],
           [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
             0.00000000e+00]])



## 计算 M_projection

### 正交视图

正交视图的可视区域是一个规整的正方体。定义一个可视区域范围：l,r(左右), t,b(上下), n,f(前后)的数据组成`[l,r] [t,b] [n,f]`长方体。然后转换到`[-1,1] [-1,1] [-1,1]` 的正方体可视区域内。要计算这个这个变换矩阵，需要先平移到原点，然后进行缩放。

平移矩阵的计算：
1. 计算长方体的中心为： $[\frac{(l+r)}{2}, \frac{(t+b)}{2}, \frac{(n+f)}{2}]$
2. 平移矩阵：将中心点平移到原点，矩阵最后一列应该为负
$$
\begin{bmatrix} 
1 & 0 & 0 & -\frac{l+r}{2} \\ 
0 & 1 & 0 & -\frac{t+b}{2} \\ 
0 & 0 & 1 & -\frac{n+f}{2} \\ 
0 & 0 & 0 & 1 
\end{bmatrix}
$$
3. 缩放矩阵：
需要长方形压缩到正方形的区域，也就是将x,y,z轴方向都缩放到`[-1,1]`范围内。使用线性变换（仿射变换）可以实现：

假设需要将点$P_{view}(x_v, y_v, z_v)$变换到$P_{proj}(x_p, y_p, z_p)$, 先对x方向进行变换：

直线公式为： $f(x) = Ax + B$, 我需要将x的范围压缩到[-1,1], 所以将(l, r)可以带入公式，列出方程：
$$
\begin{cases}
A_x * l + B_x = -1 \\
A_x * r + B_x = 1
\end{cases}
$$
求解出，$A_x=\frac{2}{r-l}$, $B_x=-\frac{r+l}{r-l}$

跟上面同样的方法，计算出y,z方向的线性公式的系数：
$$
A_y=\frac{2}{t-b}, B_y=-\frac{t+b}{t-b} \\
A_z=\frac{2}{f-n}, B_z=-\frac{f+n}{f-n}
$$

变换矩阵一般由旋转，缩放，平移的形式组成。但对于目前的变换中没有旋转。可以分两种方式表示，第一种：旋转矩阵乘平移矩阵的到最终的变换矩阵。第二种是直接写变换矩阵。
$$

第一种：

平移矩阵$T$在本章将的前部分已经给出：
T = 
\begin{bmatrix} 
1 & 0 & 0 & -\frac{l+r}{2} \\ 
0 & 1 & 0 & -\frac{t+b}{2} \\ 
0 & 0 & 1 & -\frac{n+f}{2} \\ 
0 & 0 & 0 & 1 
\end{bmatrix}
$$

旋转矩阵R由将前面线性计算的A部分组成
$$
R = 
\begin{bmatrix} 
A_x & 0 & 0 & 0 \\ 
0 & A_y & 0 & 0 \\ 
0 & 0 & A_z & 0 \\ 
0 & 0 & 0 & 1 
\end{bmatrix}
=
\begin{bmatrix} 
\frac{2}{r-l} & 0 & 0 & 0 \\ 
0 & \frac{2}{r-b} & 0 & 0 \\ 
0 & 0 & \frac{2}{f-n} & 0 \\ 
0 & 0 & 0 & 1 
\end{bmatrix}
$$

最终的变换矩阵 $M = R * T$， 得到：
$$
\begin{bmatrix} 
\frac{2}{r-l} & 0 & 0 & -\frac{r+l}{r-l} \\ 
0 & \frac{2}{r-b} & 0 & -\frac{t+b}{t-b} \\ 
0 & 0 & \frac{2}{f-n} & -\frac{f+n}{f-n} \\ 
0 & 0 & 0 & 1 
\end{bmatrix}
$$

**第二种方法：**

根据前面线性计算出来的$A_x, B_x, A_y, B_y, A_z, B_z$的内容直接带入到变换矩阵中：
$$
\begin{bmatrix} 
A & B \\ 
0 & 1 
\end{bmatrix}
$$

也可以得到M矩阵：$
\begin{bmatrix} 
\frac{2}{r-l} & 0 & 0 & -\frac{r+l}{r-l} \\ 
0 & \frac{2}{r-b} & 0 & -\frac{t+b}{t-b} \\ 
0 & 0 & \frac{2}{f-n} & -\frac{f+n}{f-n} \\ 
0 & 0 & 0 & 1 
\end{bmatrix}
$


## 三角形光栅化

## 重心坐标

## 隐藏面剔除

## 简易摄像机控制

## 优化摄像机

## 着色

## 更多数据

## 切线空间法向量映射

## 阴影映射

## 环境光遮蔽

## (附加)卡通渲染

## tinycompiler

## tinyfloat

## tinyoptimizer
