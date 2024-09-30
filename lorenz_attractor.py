import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

# Lorenz parameters
sigma = 10
rho = 28
beta = 8/3

def lorenz(w, t):
    x, y, z = w
    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ])

# Initial state
state0 = [1.0, 1.0, 1.0]
# Time points
t = np.arange(0, 50, 0.01)

# Solve ODE
sol = odeint(lorenz, state0, t)

# Setting up the figure for 3D plotting
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Initial line and point objects
line, = ax.plot([], [], [], lw=0.5)
point, = ax.plot([], [], [], 'ro')

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('Lorenz Attractor')

# View initialization
ax.view_init(elev=20., azim=30)

# Add a text box with the Lorenz equations
equation_text = r'$\frac{dx}{dt} = \sigma(y - x)$' + "\n" + \
                r'$\frac{dy}{dt} = x(\rho - z) - y$' + "\n" + \
                r'$\frac{dz}{dt} = xy - \beta z$'
ax.text2D(0.05, 0.95, equation_text, transform=ax.transAxes, fontsize=10, 
          verticalalignment='top', bbox=dict(boxstyle="round", facecolor='white', alpha=0.7))

# Data for animation
x, y, z = sol[:10, 0].tolist(), sol[:10, 1].tolist(), sol[:10, 2].tolist()  # Initial 10 points

def init():
    line.set_data(x, y)
    line.set_3d_properties(z)
    point.set_data(x[-1:], y[-1:])
    point.set_3d_properties(z[-1:])
    return line, point

def animate(i):
    if i < 10:  # Skip the first 10 frames for animation
        return line, point
    
    x.append(sol[i, 0])
    y.append(sol[i, 1])
    z.append(sol[i, 2])
    
    line.set_data(x, y)
    line.set_3d_properties(z)
    
    point.set_data(x[-1:], y[-1:])
    point.set_3d_properties(z[-1:])
    
    ax.set_xlim3d([min(x), max(x)])
    ax.set_ylim3d([min(y), max(y)])
    ax.set_zlim3d([min(z), max(z)])
    
    return line, point

anim = FuncAnimation(fig, animate, init_func=init, frames=len(t), interval=20, blit=False)

plt.show()