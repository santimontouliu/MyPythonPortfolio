# Comparison between the RK4 method and Euler's method to solve the duffing oscillator simulation problem.
# The RK4 method will generally provide a more accurate solution than the Euler method, especially for systems with rapid changes or over longer 
# integration times. We should see that the RK4 solution remains stable or follows the expected chaotic or periodic trajectory more accurately.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters for the Duffing Oscillator
delta = 0.3
alpha = -1
beta = 1
gamma = 0.3
omega = 1.2

# Time parameters
dt = 0.01
t = np.arange(0, 200, dt)

# Initial conditions
y0 = [1, 0]  # [position, velocity]

def duffing_oscillator(t, y):
    x, v = y
    a = gamma * np.cos(omega * t)
    dvdt = -delta * v - alpha * x - beta * x**3 + a
    dxdt = v
    return [dxdt, dvdt]

# Runge-Kutta 4th order method
def rk4(f, t, y0, dt):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(len(t)-1):
        k1 = np.array(f(t[i], y[i]))
        k2 = np.array(f(t[i] + 0.5*dt, y[i] + 0.5*k1*dt))
        k3 = np.array(f(t[i] + 0.5*dt, y[i] + 0.5*k2*dt))
        k4 = np.array(f(t[i] + dt, y[i] + k3*dt))
        # Here, we use numpy element-wise multiplication
        y[i+1] = y[i] + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return y

# Solve
solution = rk4(duffing_oscillator, t, y0, dt)
x, v = solution[:, 0], solution[:, 1]

# Setting up the plot with equations
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-1.5, 1.5)
ax.set_title('Duffing Oscillator (RK4 vs Euler)')
ax.set_xlabel('Position ($x$)')
ax.set_ylabel('Velocity ($v$)')

eq_text = ax.text(0.02, 0.05, r'$\ddot{x} + \delta \dot{x} + \alpha x + \beta x^3 = \gamma \cos(\omega t)$', 
                  transform=ax.transAxes, fontsize=12)

rk4_line, = ax.plot([], [], 'ro-', lw=1, label='RK4')
euler_line, = ax.plot([], [], 'b--', lw=1, label='Euler')  # Placeholder for Euler
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.legend()

# Here's a simple Euler implementation for comparison
def euler_method():
    x_e, v_e = np.zeros(len(t)), np.zeros(len(t))
    x_e[0], v_e[0] = y0
    for i in range(1, len(t)):
        dx, dv = duffing_oscillator(t[i-1], [x_e[i-1], v_e[i-1]])
        x_e[i] = x_e[i-1] + dx * dt
        v_e[i] = v_e[i-1] + dv * dt
    return x_e, v_e

x_e, v_e = euler_method()

def animate(i):
    rk4_line.set_data(x[:i], v[:i])
    euler_line.set_data(x_e[:i], v_e[:i])
    time_text.set_text(f'Time = {t[i]:.2f}')
    return [rk4_line, euler_line, time_text]

anim = FuncAnimation(fig, animate, frames=len(t), interval=20, blit=True)
plt.show()