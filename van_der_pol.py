import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
from scipy import interpolate

# Parameter for the Van der Pol oscillator
mu = 2.0

def van_der_pol(y, t):
    x, v = y
    dxdt = v
    dvdt = mu * (1 - x**2) * v - x
    return [dxdt, dvdt]

# Initial conditions
y0 = [0.1, 0.0]

# Time vector
t = np.linspace(0, 100, 5000)  # Increased from 1000 to 5000 for example

# Solve ODE
sol = odeint(van_der_pol, y0, t)

# Set up the figure, the axis, and the plot elements we want to animate
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Create interpolation functions
f_x = interpolate.interp1d(t, sol[:, 0], kind='cubic')
f_v = interpolate.interp1d(t, sol[:, 1], kind='cubic')

# Phase space plot
line_phase, = ax1.plot([], [], 'b-', lw=1)
point_phase, = ax1.plot([], [], 'ro', markersize=3)
ax1.set_xlabel('x')
ax1.set_ylabel('dx/dt')
ax1.set_title('Phase Space of Van der Pol Oscillator')
ax1.set_xlim(-5, 5)
ax1.set_ylim(-5, 5)

# Adding the equation as a text to the phase space plot
equation_text = r'$\ddot{x} - \mu(1 - x^2)\dot{x} + x = 0$'
ax1.text(0.05, 0.95, equation_text, transform=ax1.transAxes, fontsize=12, 
         verticalalignment='top', bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))

# Time series plot
line_time, = ax2.plot([], [], 'g-', lw=1)
ax2.set_xlabel('Time')
ax2.set_ylabel('x(t)')
ax2.set_xlim(0, 100)
ax2.set_ylim(-3, 3)

def init():
    line_phase.set_data([], [])
    point_phase.set_data([], [])
    line_time.set_data([], [])
    return line_phase, point_phase, line_time

def animate(i):
    if i >= len(t):  # Ensure i does not exceed the length of t (and sol)
        return line_phase, point_phase, line_time
    
    # Phase space data update
    line_phase.set_data(sol[:i+1, 0], sol[:i+1, 1])  # Use i+1 to ensure we always pass a sequence
    point_phase.set_data([sol[i, 0]], [sol[i, 1]])  # Wrap in lists to ensure we're passing sequences
    
    # Time series data update
    line_time.set_data(t[:i+1], sol[:i+1, 0])  # Similarly, ensure we're passing sequences
    
    return line_phase, point_phase, line_time

# Animation creation
anim = FuncAnimation(fig, animate, init_func=init, frames=len(t), interval=5, blit=True)  # Reduced interval for smoother animation


plt.tight_layout()
plt.show()