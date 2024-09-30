import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters for the Duffing Oscillator
delta = 0.3  # Damping coefficient
alpha = -1   # Linear stiffness (negative for bistability)
beta = 1     # Nonlinear stiffness
gamma = 0.3  # Driving force amplitude
omega = 1.2  # Driving force frequency

# Time parameters
dt = 0.01
t = np.arange(0, 200, dt)

# Initial conditions
x0, v0 = 1, 0  # Initial position and velocity

# Duffing oscillator equations
def duffing_oscillator(x, v, t):
    a = gamma * np.cos(omega * t)
    dvdt = -delta * v - alpha * x - beta * x**3 + a
    dxdt = v
    return dxdt, dvdt

# Solve using numerical integration (Euler method for simplicity)
x, v = np.zeros(len(t)), np.zeros(len(t))
x[0], v[0] = x0, v0

for i in range(1, len(t)):
    dx, dv = duffing_oscillator(x[i-1], v[i-1], t[i-1])
    x[i] = x[i-1] + dx * dt
    v[i] = v[i-1] + dv * dt

# Setting up the plot
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-1.5, 1.5)
ax.set_title('Duffing Oscillator')
ax.set_xlabel('Position ($x$)')
ax.set_ylabel('Velocity ($v$)')

eq_text = ax.text(0.02, 0.05, r'$\ddot{x} + \delta \dot{x} + \alpha x + \beta x^3 = \gamma \cos(\omega t)$', 
                  transform=ax.transAxes, fontsize=12)

line, = ax.plot([], [], 'ro-', lw=1)  # lw=1 to make it half the width of what was before
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def animate(i):
    line.set_data(x[:i], v[:i])
    time_text.set_text(f'Time = {t[i]:.2f}')
    return line, time_text

anim = FuncAnimation(fig, animate, init_func=init, frames=len(t), interval=20, blit=True)

plt.show()
