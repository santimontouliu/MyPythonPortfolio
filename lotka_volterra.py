import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

# Parameters
alpha = 1.0   # Prey growth rate
beta = 0.02   # Prey death rate per predator
delta = 0.01  # Predator growth rate per prey eaten
gamma = 0.5   # Predator death rate

# Initial conditions
x0, y0 = 50, 10  # Initial population of prey and predators

# Time setup
dt = 0.01
t = np.arange(0, 100, dt)

def lotka_volterra(state, t):
    x, y = state  # Unpack here instead of function parameters
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]  # Return the derivatives as a list

# Solve ODE
solution = odeint(lotka_volterra, [x0, y0], t)

x, y = solution[:, 0], solution[:, 1]

# Plotting setup
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
line_prey, = ax[0].plot([], [], 'g-', label='Prey')
line_pred, = ax[0].plot([], [], 'r-', label='Predator')
ax[0].legend()
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Population')
ax[0].set_xlim(0, t.max())
ax[0].set_ylim(0, max(max(x), max(y))*1.1)

phase_line, = ax[1].plot([], [], 'b-', label='Phase trajectory')
phase_tip, = ax[1].plot([], [], 'o', color='orange', markersize=5)  # Marker for the trajectory tip
ax[1].set_xlabel('Prey')
ax[1].set_ylabel('Predator')
ax[1].set_xlim(0, max(x)*1.1)
ax[1].set_ylim(0, max(y)*1.1)
ax[1].legend()

time_text = ax[0].text(0.02, 0.95, '', transform=ax[0].transAxes)

def init():
    phase_tip.set_data([], [])
    return line_prey, line_pred, phase_line, phase_tip, time_text


def animate(i):
    line_prey.set_data(t[:i], x[:i])
    line_pred.set_data(t[:i], y[:i])
    phase_line.set_data(x[:i], y[:i])
    # Update the tip position with lists containing current x and y
    phase_tip.set_data([x[i]], [y[i]])  
    time_text.set_text(f'Time = {t[i]:.2f}')
    return line_prey, line_pred, phase_line, phase_tip, time_text

anim = FuncAnimation(fig, animate, init_func=init, frames=len(t), interval=20, blit=True)

plt.tight_layout()
plt.show()