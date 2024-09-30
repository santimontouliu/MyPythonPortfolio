import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Define the system parameters
m1, m2 = 1.0, 1.0  # Masses
k1, k2 = 1.0, 1.5  # Spring constants for the outer springs
kc = 0.5  # Coupling spring constant
L1, L2, Lc = 1.0, 1.0, 0.5  # Natural lengths of the springs

# Define the system of ODEs
def coupled_oscillator(t, y, m1, m2, k1, k2, kc):
    x1, v1, x2, v2 = y
    a1 = (-k1 * (x1 - L1) + kc * (x2 - x1 - Lc)) / m1
    a2 = (-k2 * (x2 - L2) - kc * (x2 - x1 - Lc)) / m2
    return [v1, a1, v2, a2]

# Initial conditions
initial_conditions = [1.5, 0, 0, 0]  # x1, v1, x2, v2

# Time span
t_span = (0, 50)
t_eval = np.linspace(0, 50, 1000)

# Solve ODE
sol = solve_ivp(coupled_oscillator, t_span, initial_conditions, t_eval=t_eval, 
                args=(m1, m2, k1, k2, kc))

# Plotting setup with animation in one figure
fig = plt.figure(figsize=(15, 10))

# Subplot for animation
ax_anim = fig.add_subplot(2, 2, 4)
ax_anim.set_xlim(-5, 5)
ax_anim.set_ylim(-1, 1)
ax_anim.set_aspect('equal')
ax_anim.axhline(y=0, color='k', lw=0.5)
ax_anim.axvline(x=-L1, color='g', ls='--', lw=1)
ax_anim.axvline(x=L2, color='m', ls='--', lw=1)

# Setup for masses and springs in animation
mass1, = ax_anim.plot([], [], 'ro', markersize=20)
mass2, = ax_anim.plot([], [], 'bo', markersize=20)
spring1, = ax_anim.plot([], [], 'k-', lw=2)
spring2, = ax_anim.plot([], [], 'k-', lw=2)
spring_c, = ax_anim.plot([], [], 'k-', lw=1)

# Subplots for x(t), v(t), a(t)
ax_x = fig.add_subplot(2, 2, 1)
ax_v = fig.add_subplot(2, 2, 2)
ax_a = fig.add_subplot(2, 2, 3)

ax_x.set_title('Position x(t)')
ax_v.set_title('Velocity v(t)')
ax_a.set_title('Acceleration a(t)')

# Initialize line objects for real-time plotting
line_x1, = ax_x.plot([], [], label='Mass 1')
line_x2, = ax_x.plot([], [], label='Mass 2')
line_v1, = ax_v.plot([], [], label='Mass 1')
line_v2, = ax_v.plot([], [], label='Mass 2')
line_a1, = ax_a.plot([], [], label='Mass 1')
line_a2, = ax_a.plot([], [], label='Mass 2')

ax_x.legend()
ax_v.legend()
ax_a.legend()

# Data for plotting
x_data1, x_data2 = [], []
v_data1, v_data2 = [], []
a_data1, a_data2 = [], []
t_data = []

def init():
    for line in [line_x1, line_x2, line_v1, line_v2, line_a1, line_a2, mass1, mass2, spring1, spring2, spring_c]:
        line.set_data([], [])
    return line_x1, line_x2, line_v1, line_v2, line_a1, line_a2, mass1, mass2, spring1, spring2, spring_c

def animate(i):
    # Update mass positions
    x1 = [sol.y[0][i] - L1]
    y1 = [0]
    x2 = [sol.y[2][i] + 3 - L2]
    y2 = [0]
    
    mass1.set_data(x1, y1)
    mass2.set_data(x2, y2)
    spring1.set_data([-L1, x1[0]], [0, 0])
    spring2.set_data([x1[0], x2[0]], [0, 0])
    spring_c.set_data([x1[0], x1[0]], [-0.1, 0.1])
    
    t_data.append(sol.t[i])
    x_data1.append(sol.y[0][i])
    x_data2.append(sol.y[2][i])
    v_data1.append(sol.y[1][i])
    v_data2.append(sol.y[3][i])
    
    # Compute acceleration for plotting
    acc1 = (-k1 * (sol.y[0][i] - L1) + kc * (sol.y[2][i] - sol.y[0][i] - Lc)) / m1
    acc2 = (-k2 * (sol.y[2][i] - L2) - kc * (sol.y[2][i] - sol.y[0][i] - Lc)) / m2
    a_data1.append(acc1)
    a_data2.append(acc2)
    
    line_x1.set_data(t_data, x_data1)
    line_x2.set_data(t_data, x_data2)
    ax_x.set_xlim(0, max(t_data))
    ax_x.set_ylim(min(x_data1 + x_data2) - 0.5, max(x_data1 + x_data2) + 0.5)
    
    line_v1.set_data(t_data, v_data1)
    line_v2.set_data(t_data, v_data2)
    ax_v.set_xlim(0, max(t_data))
    ax_v.set_ylim(min(v_data1 + v_data2) - 0.5, max(v_data1 + v_data2) + 0.5)
    
    line_a1.set_data(t_data, a_data1)
    line_a2.set_data(t_data, a_data2)
    ax_a.set_xlim(0, max(t_data))
    ax_a.set_ylim(min(a_data1 + a_data2) - 0.5, max(a_data1 + a_data2) + 0.5)
    
    return line_x1, line_x2, line_v1, line_v2, line_a1, line_a2, mass1, mass2, spring1, spring2, spring_c

anim = FuncAnimation(fig, animate, frames=len(sol.t), init_func=init, blit=True, interval=50, repeat=False)

plt.tight_layout()
plt.show()