# 物理模型详解

## 运动方程

乒乓球的运动遵循经典力学和流体力学的规律。球体受到重力、空气阻力和马格努斯力的作用：

```math
m \frac{d\mathbf{v}}{dt} = m\mathbf{g} + \mathbf{F}_{drag} + \mathbf{F}_{magnus}
```

### 重力项
```math
\mathbf{F}_g = m\mathbf{g} = m\begin{pmatrix} 0 \\ 0 \\ -9.81 \end{pmatrix}
```

### 空气阻力
空气阻力遵循平方阻力定律：
```math
\mathbf{F}_{drag} = -\frac{1}{2} \rho v^2 S C_d \hat{\mathbf{v}}
```

其中：
- ρ = 1.225 kg/m³ (空气密度)
- S = πR² (迎风面积)
- C_d = 0.40 (阻力系数)
- v = |v| (速度大小)

### 马格努斯效应
旋转球体产生的升力：
```math
\mathbf{F}_{magnus} = \rho R^3 C_\Omega (\boldsymbol{\omega} \times \mathbf{v})
```

其中：
- C_Ω = 0.20 (马格努斯系数)
- ω 为角速度向量

## 数值积分方法

### RK4 (四阶龙格-库塔) 方法

使用四阶龙格-库塔方法进行数值积分，保证计算精度：

```python
def rk4_step(state, dt):
    k1 = f(t_n, y_n)
    k2 = f(t_n + dt/2, y_n + k1*dt/2)
    k3 = f(t_n + dt/2, y_n + k2*dt/2)
    k4 = f(t_n + dt, y_n + k3*dt)

    y_{n+1} = y_n + (dt/6)(k1 + 2k2 + 2k3 + k4)
```

## 碰撞检测和响应

### 平面碰撞模型

球体与平面表面的碰撞使用脉冲-动量法：

**碰撞检测**:
```python
# 距离计算
d = \dot{\mathbf{n}} \cdot (\mathbf{x}_{ball} - \mathbf{x}_{plane}) - R

# 相对速度
v_rel = v_ball - v_surface
v_rel_n = \dot{\mathbf{n}} \cdot v_rel

# 碰撞条件
if d ≤ 0 and v_rel_n < 0:
    # 发生碰撞
```

**碰撞响应**:
```python
# 法向冲量
J_n = -(1 + e) m v_rel_n

# 切向冲量 (Coulomb摩擦)
u = v_rel + ω × r  # 接触点相对速度
u_n = (u · n) n
u_t = u - u_n

if |u_t| ≤ μ |J_n|:
    J_t = -m u_t  # 粘着
else:
    J_t = -μ |J_n| * u_t / |u_t|  # 滑动
```

### 球拍碰撞

球拍建模为圆形平面，使用相同的碰撞框架但考虑球拍的运动状态。

## 球网检测

### 轨迹穿越检测

检测球体轨迹是否穿越球网平面：

```python
def check_net_collision(pos_old, pos_new, net, table_height):
    # 检查x方向穿越
    if pos_old.x * pos_new.x > 0:
        return False

    # 插值计算穿越点
    t_cross = -pos_old.x / (pos_new.x - pos_old.x)
    cross_pos = pos_old + t_cross * (pos_new - pos_old)

    # 检查y范围
    if abs(cross_pos.y) > net.length/2:
        return NET_CROSS_SUCCESS

    # 检查高度
    net_top = table_height + net.height
    ball_bottom = cross_pos.z - BALL_RADIUS

    if ball_bottom < net_top:
        return NET_HIT if ball_bottom > table_height else NET_CROSS_FAIL
    else:
        return NET_CROSS_SUCCESS
```

## 材料属性

### 球桌材质
- **恢复系数**: e = 0.90
- **摩擦系数**: μ = 0.25
- **表面特性**: 均匀平面，无粗糙度变化

### 球拍胶皮类型

| 类型 | 恢复系数 | 摩擦系数 | 特点 |
|------|----------|----------|------|
| 反胶(INVERTED) | 0.88 | 0.45 | 高旋转，进攻性 |
| 生胶(PIMPLED) | 0.85 | 0.30 | 中等旋转，控制性 |
| 防弧(ANTISPIN) | 0.80 | 0.15 | 低旋转，防守性 |

## 验证与校准

### 解析解对比
对于无旋转情况，轨迹方程为：
```math
y = x \tan\theta - \frac{g x^2}{2 v_0^2 \cos^2\theta}
```

### 参数敏感性分析
- **阻力系数**: 影响轨迹弧度
- **马格努斯系数**: 影响旋转效应的强度
- **碰撞参数**: 影响反弹特性和旋转传递

### 实验数据对比
系统参数基于ITTF官方标准和公开的科学文献，确保物理模型的准确性。
