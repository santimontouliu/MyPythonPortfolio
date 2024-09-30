import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# Constants
q = 1.6e-19  # Charge of particle (C)
m = 9.1e-31  # Mass of electron (kg)
B = 1.0e-3  # Magnetic field strength (T)
E = 1e2  # Electric field in the gap (V/m), simplified for acceleration
v0 = 1e7  # Initial velocity (m/s)
r_max = 0.5  # Maximum radius to simulate (m)
dt = 1e-11  # Time step (s)

# Initial conditions
theta = 0
r = m * v0 / (q * B)
v = v0
x, y = r * np.cos(theta), r * np.sin(theta)

# Simulation parameters
positions = []
time = []

def update_position():
    global x, y, v, theta, r
    # Simplified acceleration in gap
    if np.pi/2 < theta < 3*np.pi/2 or -np.pi/2 < theta < np.pi/2:
        v += q * E * dt / m
    
    # Update velocity direction due to magnetic field
    theta += q * B * dt / m
    
    # Update radius with new velocity
    r = m * v / (q * B)
    if r > r_max:
        return False
    
    x, y = r * np.cos(theta), r * np.sin(theta)
    positions.append((x, y))
    time.append(time[-1] + dt if time else 0)
    return True

# Setting up the plot
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-r_max*1.1, r_max*1.1)
ax.set_ylim(-r_max*1.1, r_max*1.1)
ax.set_aspect('equal')
ax.set_title('Cyclotron Motion with Field Vectors')

# Adding D's for visualization
d1 = Rectangle((-r_max/2, -0.05), r_max, 0.1, edgecolor='gray', facecolor='none', lw=2)
d2 = Rectangle((-r_max/2, 0.05), r_max, 0.1, edgecolor='gray', facecolor='none', lw=2)
ax.add_patch(d1)
ax.add_patch(d2)

# Vector for B (Magnetic Field) - pointing into the page, so we'll use a symbol or text
ax.text(0, r_max*0.9, r'$\mathbf{B}$', fontsize=20, ha='center', va='center')

# Vector for E (Electric Field) - we'll represent this with an arrow since E can be in different directions
E_arrow = ax.arrow(0, 0, 0, 0.1, head_width=r_max*0.05, head_length=r_max*0.05, fc='r', ec='r')

line, = ax.plot([], [], 'r-', lw=1)
point, = ax.plot([], [], 'bo', markersize=3)

# Equations as text
equation_text = r'$F = q(v \times B)$' + "\n" + \
                r'$\omega = \frac{qB}{m}$' + "\n" + \
                r'$r = \frac{mv}{qB}$'
ax.text(0.05, 0.95, equation_text, transform=ax.transAxes, fontsize=10, 
        verticalalignment='top', bbox=dict(boxstyle="round", facecolor='white', alpha=0.7))

def init():
    line.set_data([], [])
    point.set_data([], [])
    return line, point

def animate(i):
    for _ in range(100):  # Adjust as needed for smoothness
        if not update_position():
            anim.event_source.stop()
            return line, point, E_arrow
    
    x_vals, y_vals = zip(*positions[-100:])  # Plot last 100 positions
    line.set_data(x_vals, y_vals)
    point.set_data([positions[-1][0]], [positions[-1][1]])

    # Update Electric Field Vector
    # This is a simplification. Normally E would switch direction in a cyclotron; here we'll just show its magnitude
    # Assuming E is in the y-direction for simplicity; typically it would oscillate
    Ey = E if np.sin(theta) > 0 else -E  # Simplified, not physically accurate but for visualization
    E_arrow.set_data(dx=0, dy=Ey*1e-6, x=0, y=0)  # Scale factor 1e-6 for visibility

    return line, point, E_arrow

anim = FuncAnimation(fig, animate, init_func=init, blit=True, frames=1000, interval=50)

plt.show()
