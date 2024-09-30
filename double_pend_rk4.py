import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
g = 9.8  # acceleration due to gravity, in m/s^2
L1, L2 = 1.0, 1.0  # length of pendulum rods in meters
M1, M2 = 1.0, 1.0  # mass of pendulum bobs in kg

# Initial conditions
theta1, theta2 = np.pi/4, 0  # initial angles in radians
dtheta1, dtheta2 = 0, 0  # initial angular velocities

# Simulation parameters
dt = 0.02  # time step, in seconds. Adjusted for RK4
t = np.arange(0.0, 20, dt)  # time array for 20 seconds

def derivatives(state, t):
    theta1, theta2, dtheta1, dtheta2 = state
    m = M1 + M2
    d1 = L1 * (2 * M1 + M2 - M2 * np.cos(2 * theta1 - 2 * theta2))
    d2 = L2 * (2 * M1 + M2 - M2 * np.cos(2 * theta1 - 2 * theta2))
    
    dtheta1_dt = (-g * (2 * M1 + M2) * np.sin(theta1) - M2 * g * np.sin(theta1 - 2 * theta2) - 
                  2 * np.sin(theta1 - theta2) * M2 * (dtheta2**2 * L2 + dtheta1**2 * L1 * np.cos(theta1 - theta2))) / d1
    
    dtheta2_dt = (2 * np.sin(theta1 - theta2) * (dtheta1**2 * L1 * m + g * m * np.cos(theta1) + 
                  dtheta2**2 * L2 * M2 * np.cos(theta1 - theta2))) / d2
    
    return np.array([dtheta1, dtheta2, dtheta1_dt, dtheta2_dt])

def rk4(state, t, dt, derivatives):
    k1 = dt * derivatives(state, t)
    k2 = dt * derivatives(state + 0.5 * k1, t + 0.5 * dt)
    k3 = dt * derivatives(state + 0.5 * k2, t + 0.5 * dt)
    k4 = dt * derivatives(state + k3, t + dt)
    return state + (1.0/6.0) * (k1 + 2*k2 + 2*k3 + k4)

# Initial state
state = np.array([theta1, theta2, dtheta1, dtheta2])
states = []

# Simulate using RK4
for time in t:
    state = rk4(state, time, dt, derivatives)
    states.append(state)

states = np.array(states)
theta1_array, theta2_array, _, _ = states.T  # Unpack only what we need for plotting

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
    line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
    trail.set_data(x2[max(0, i-50):i], y2[max(0, i-50):i])
    return line, trail

anim = FuncAnimation(fig, animate, frames=len(t), init_func=init, interval=dt*1000, blit=True)

plt.title('Double Pendulum Animation using RK4')
plt.show()

