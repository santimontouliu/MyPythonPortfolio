import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
from PIL import Image  # Import Pillow's Image module for saving as GIF

# Gravitational constant
G = 6.67430e-11   # Temporarily increase G by a factor for clearer effect

# Masses of the three bodies (in kg)
m1, m2, m3 = 1e30, 1e30, 1e30  # Example: much larger masses, like small stars or large asteroids

# Initial positions forming an equilateral triangle
r1 = np.array([1e11, 0.0])
r2 = np.array([-0.5e11, 0.866e11])  # 0.866 = sqrt(3)/2 for equilateral triangle
r3 = np.array([-0.5e11, -0.866e11])

# Initial velocities for circular orbits around the center of mass
v1, v2, v3 = np.array([0.0, 10e3]), np.array([5e3, 0.0]), np.array([-5e3, -5e3])

# Combine into one state vector
state = np.concatenate([r1, r2, r3, v1, v2, v3])

def three_body_equations(w, t, G, m1, m2, m3):
    # Unpack state
    r1, r2, r3, v1, v2, v3 = w.reshape(6,2)
    
    # Distance vectors
    r12 = r2 - r1
    r13 = r3 - r1
    r23 = r3 - r2

    # Gravitational acceleration
    a1 = G * (m2 * r12 / np.linalg.norm(r12)**3 + m3 * r13 / np.linalg.norm(r13)**3)
    a2 = G * (m1 * -r12 / np.linalg.norm(r12)**3 + m3 * r23 / np.linalg.norm(r23)**3)
    a3 = G * (m1 * -r13 / np.linalg.norm(r13)**3 + m2 * -r23 / np.linalg.norm(r23)**3)

    # Stack velocities and accelerations
    return np.vstack((v1, v2, v3, a1, a2, a3)).flatten()

# Time vector
t = np.linspace(0, 3.154e8, 5000)  # More points might help in capturing subtle movements

# Solve ODE
sol = odeint(three_body_equations, state.flatten(), t, args=(G, m1, m2, m3))

fig, ax = plt.subplots(figsize=(10, 10))
line1, = ax.plot([], [], 'o-', lw=2, color='blue', label='Body 1')
line2, = ax.plot([], [], 'o-', lw=2, color='red', label='Body 2')
line3, = ax.plot([], [], 'o-', lw=2, color='green', label='Body 3')
ax.set_xlim(-3e11, 3e11)
ax.set_ylim(-3e11, 3e11)
ax.set_aspect('equal')
ax.legend()

# Before the animation setup, initialize empty lists for path accumulation:
x1, y1 = [], []
x2, y2 = [], []
x3, y3 = [], []

def animate(i):
    # Append new positions to our lists
    x1.append(sol[i, 0])
    y1.append(sol[i, 1])
    x2.append(sol[i, 2])
    y2.append(sol[i, 3])
    x3.append(sol[i, 4])
    y3.append(sol[i, 5])
    
    # Update paths for all three bodies
    line1.set_data(x1, y1)
    line2.set_data(x2, y2)
    line3.set_data(x3, y3)
    
    # Adjust the axes to keep all points in view
    ax.set_xlim(min(x1+x2+x3) - 1e10, max(x1+x2+x3) + 1e10)
    ax.set_ylim(min(y1+y2+y3) - 1e10, max(y1+y2+y3) + 1e10)
    
    return line1, line2, line3

anim = FuncAnimation(fig, animate, frames=len(t), interval=50, blit=True)

plt.title('Three-Body Problem Simulation')
plt.xlabel('X position (m)')
plt.ylabel('Y position (m)')
plt.grid(True)

plt.show()