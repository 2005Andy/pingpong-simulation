# Physics Model Details

## Equations of Motion

Ping-pong ball motion follows the laws of classical mechanics and fluid dynamics. The ball is subject to gravity, air drag, and Magnus force:

```math
m \frac{d\mathbf{v}}{dt} = m\mathbf{g} + \mathbf{F}_{drag} + \mathbf{F}_{magnus}
```

### Gravity Term
```math
\mathbf{F}_g = m\mathbf{g} = m\begin{pmatrix} 0 \\ 0 \\ -9.81 \end{pmatrix}
```

### Air Drag
Air drag follows the quadratic drag law:
```math
\mathbf{F}_{drag} = -\frac{1}{2} \rho v^2 S C_d \hat{\mathbf{v}}
```

Where:
- ρ = 1.225 kg/m³ (air density)
- S = πR² (frontal area)
- C_d = 0.40 (drag coefficient)
- v = |v| (speed magnitude)

### Magnus Effect
Lift force produced by spinning ball:
```math
\mathbf{F}_{magnus} = \rho R^3 C_\Omega (\boldsymbol{\omega} \times \mathbf{v})
```

Where:
- C_Ω = 0.20 (Magnus coefficient)
- ω is the angular velocity vector

## Numerical Integration Methods

### RK4 (4th-order Runge-Kutta) Method

Using 4th-order Runge-Kutta method for numerical integration to ensure calculation accuracy:

```python
def rk4_step(state, dt):
    k1 = f(t_n, y_n)
    k2 = f(t_n + dt/2, y_n + k1*dt/2)
    k3 = f(t_n + dt/2, y_n + k2*dt/2)
    k4 = f(t_n + dt, y_n + k3*dt)

    y_{n+1} = y_n + (dt/6)(k1 + 2k2 + 2k3 + k4)
```

## Collision Detection and Response

### Plane Collision Model

Ball collision with planar surfaces uses the impulse-momentum method:

**Collision Detection**:
```python
# Distance calculation
d = \dot{\mathbf{n}} \cdot (\mathbf{x}_{ball} - \mathbf{x}_{plane}) - R

# Relative velocity
v_rel = v_ball - v_surface
v_rel_n = \dot{\mathbf{n}} \cdot v_rel

# Collision condition
if d ≤ 0 and v_rel_n < 0:
    # Collision occurred
```

**Collision Response**:
```python
# Normal impulse
J_n = -(1 + e) m v_rel_n

# Tangential impulse (Coulomb friction)
u = v_rel + ω × r  # Relative velocity at contact point
u_n = (u · n) n
u_t = u - u_n

if |u_t| ≤ μ |J_n|:
    J_t = -m u_t  # Sticking
else:
    J_t = -μ |J_n| * u_t / |u_t|  # Sliding
```

### Racket Collision

Racket is modeled as a circular plane, using the same collision framework but considering racket motion state.

## Net Detection

### Trajectory Crossing Detection

Detect if ball trajectory crosses the net plane:

```python
def check_net_collision(pos_old, pos_new, net, table_height):
    # Check x-direction crossing
    if pos_old.x * pos_new.x > 0:
        return False

    # Interpolate crossing point
    t_cross = -pos_old.x / (pos_new.x - pos_old.x)
    cross_pos = pos_old + t_cross * (pos_new - pos_old)

    # Check y extent
    if abs(cross_pos.y) > net.length/2:
        return NET_CROSS_SUCCESS

    # Check height
    net_top = table_height + net.height
    ball_bottom = cross_pos.z - BALL_RADIUS

    if ball_bottom < net_top:
        return NET_HIT if ball_bottom > table_height else NET_CROSS_FAIL
    else:
        return NET_CROSS_SUCCESS
```

## Material Properties

### Table Material
- **Restitution coefficient**: e = 0.90
- **Friction coefficient**: μ = 0.25
- **Surface characteristics**: Uniform plane, no roughness variation

### Racket Rubber Types

| Type | Restitution | Friction | Characteristics |
|------|-------------|----------|------------------|
| Inverted | 0.88 | 0.45 | High spin, offensive |
| Pimpled | 0.85 | 0.30 | Medium spin, control |
| Anti-spin | 0.80 | 0.15 | Low spin, defensive |

## Validation and Calibration

### Analytical Solution Comparison
For non-spinning case, trajectory equation is:
```math
y = x \tan\theta - \frac{g x^2}{2 v_0^2 \cos^2\theta}
```

### Parameter Sensitivity Analysis
- **Drag coefficient**: Affects trajectory curvature
- **Magnus coefficient**: Affects spin effect strength
- **Collision parameters**: Affects bounce characteristics and spin transfer

### Experimental Data Comparison
System parameters are based on ITTF official standards and published scientific literature to ensure physics model accuracy.
