import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
g = 9.8  # acceleration due to gravity, in m/s^2
L1, L2 = 1.0, 1.0  # length of pendulum rods in meters
M1, M2 = 2.5, 1.5  # mass of pendulum bobs in kg

# Initial conditions
theta1, theta2 = 2*np.pi/3, 0  # initial angles in radians
dtheta1, dtheta2 = 0, 1  # initial angular velocities

# Simulation parameters
dt = 0.04  # time step, in seconds
t = np.arange(0.0, 20, dt)  # time array for 20 seconds

# Storage for angles over time
theta1_array = np.zeros_like(t)
theta2_array = np.zeros_like(t)

# Initial state
theta1_array[0] = theta1
theta2_array[0] = theta2

# Simulate the double pendulum
for i in range(1, len(t)):
    # Here, we use simple forward Euler method for simplicity. 
    # Note: In a real physics simulation, we could use something like RK4 for accuracy.
    
    # These are the equations of motion for the double pendulum:
    num1 = -g * (2 * M1 + M2) * np.sin(theta1_array[i-1]) - M2 * g * np.sin(theta1_array[i-1] - 2 * theta2_array[i-1]) - 2 * np.sin(theta1_array[i-1] - theta2_array[i-1]) * M2 * (dtheta2**2 * L2 + dtheta1**2 * L1 * np.cos(theta1_array[i-1] - theta2_array[i-1]))
    den1 = L1 * (2 * M1 + M2 - M2 * np.cos(2 * theta1_array[i-1] - 2 * theta2_array[i-1]))
    dtheta1_dt = num1 / den1

    num2 = 2 * np.sin(theta1_array[i-1] - theta2_array[i-1]) * (dtheta1**2 * L1 * (M1 + M2) + g * (M1 + M2) * np.cos(theta1_array[i-1]) + dtheta2**2 * L2 * M2 * np.cos(theta1_array[i-1] - theta2_array[i-1]))
    den2 = L2 * (2 * M1 + M2 - M2 * np.cos(2 * theta1_array[i-1] - 2 * theta2_array[i-1]))
    dtheta2_dt = num2 / den2

    dtheta1 += dtheta1_dt * dt
    dtheta2 += dtheta2_dt * dt

    theta1_array[i] = theta1_array[i-1] + dtheta1 * dt
    theta2_array[i] = theta2_array[i-1] + dtheta2 * dt

# Calculate positions
x1 = L1 * np.sin(theta1_array)
y1 = -L1 * np.cos(theta1_array)
x2 = x1 + L2 * np.sin(theta2_array)
y2 = y1 - L2 * np.cos(theta2_array)

# Animation setup
fig, ax = plt.subplots()
ax.set_xlim((-2, 2))
ax.set_ylim((-2, 1))
line, = ax.plot([], [], 'o-', lw=2)
trail, = ax.plot([], [], '-', lw=1, color='grey')

def init():
    line.set_data([], [])
    trail.set_data([], [])
    return line, trail

def animate(i):
    line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])  # First point is pivot, second is first bob, third is second bob
    trail.set_data(x2[max(0, i-50):i], y2[max(0, i-50):i])  # Trail of the second bob
    return line, trail

anim = FuncAnimation(fig, animate, frames=len(t), init_func=init, interval=dt*1000, blit=True)

plt.title('Double Pendulum Animation')
plt.show()