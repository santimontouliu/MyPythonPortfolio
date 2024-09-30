import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters for both simulations
N = 32  # Number of oscillators
n_steps = 10000
dt = 0.01

# Initial conditions (first mode excited)
x_initial = np.sin(np.pi * np.arange(1, N+1) / (N + 1))
v_initial = np.zeros(N)

def fput(alpha, x, v):
    a = np.zeros_like(x)
    a[0] = (x[1] - 2*x[0]) + alpha * ((x[1] - x[0])**3)
    a[-1] = (x[-2] - 2*x[-1]) + alpha * ((x[-2]**3 - (x[-1] - x[-2])**3))
    a[1:-1] = (x[2:] - 2*x[1:-1] + x[:-2]) + alpha * ((x[2:] - x[1:-1])**3 - (x[1:-1] - x[:-2])**3)
    return v, a

def verlet_step(x, v, alpha, dt):
    v_half = v + 0.5 * dt * fput(alpha, x, v)[1]
    x_new = x + dt * v_half
    v_new = v_half + 0.5 * dt * fput(alpha, x_new, v_half)[1]
    return x_new, v_new

# Nonlinear simulation
alpha_nonlinear = 0.25
X_nonlinear = [x_initial]
V_nonlinear = [v_initial]

# Linear simulation (alpha set to 0 for linear behavior)
alpha_linear = 0
X_linear = [x_initial]
V_linear = [v_initial]

for _ in range(n_steps):
    X_nonlinear.append(verlet_step(X_nonlinear[-1], V_nonlinear[-1], alpha_nonlinear, dt)[0])
    V_nonlinear.append(verlet_step(X_nonlinear[-1], V_nonlinear[-1], alpha_nonlinear, dt)[1])
    X_linear.append(verlet_step(X_linear[-1], V_linear[-1], alpha_linear, dt)[0])
    V_linear.append(verlet_step(X_linear[-1], V_linear[-1], alpha_linear, dt)[1])

X_nonlinear = np.array(X_nonlinear)
X_linear = np.array(X_linear)

# Animation setup
fig, ax = plt.subplots()
line_nonlinear, = ax.plot([], [], 'b-', lw=2, label='Nonlinear')
line_linear, = ax.plot([], [], 'r--', lw=2, label='Linear')
ax.set_xlim(0, N)
ax.set_ylim(-1.5, 1.5)
ax.set_title('FPU Problem: Linear vs Nonlinear')
ax.set_xlabel('Oscillator Index')
ax.set_ylabel('Displacement')
ax.legend()

def init():
    line_nonlinear.set_data([], [])
    line_linear.set_data([], [])
    return line_nonlinear, line_linear

def animate(i):
    line_nonlinear.set_data(np.arange(N), X_nonlinear[i])
    line_linear.set_data(np.arange(N), X_linear[i])
    return line_nonlinear, line_linear

anim = FuncAnimation(fig, animate, init_func=init, frames=n_steps, interval=10, blit=True)

plt.show()