import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# if you want the plots to look comic-style, uncomment the next line! 
# plt.xkcd() 

# Parameters for the harmonic oscillator
m = 1.0  # mass in kg
k = 1.0  # spring constant in N/m
omega = np.sqrt(k / m)  # natural frequency

# Initial conditions
A = 1.0  # amplitude in meters
x0 = A  # initial position
v0 = 0.0  # initial velocity

# Time parameters
dt = 0.05
t = np.arange(0, 20, dt)  # simulate for 20 seconds

# Calculate the position over time using the analytical solution for simplicity
x = A * np.cos(omega * t)
v = -A * omega * np.sin(omega * t)  # velocity is the derivative of position
ac = -A * omega**2 * np.cos(omega * t) # acceleration is the derivative of velocity

# Setup the figure, the axis, and the plot elements we want to animate
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 15))

# Setup the oscillator animation in ax1
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax1.set_aspect('equal')
ax1.set_title('Harmonic Oscillator')

# Mass on a spring
spring, = ax1.plot([], [], 'k-', lw=2)
mass, = ax1.plot([], [], 'o', markersize=10)
anchor, = ax1.plot(0, 0, 'k^', markersize=10)

# Setup the position vs time plot in ax2
ax2.set_xlim(0, t[-1])
ax2.set_ylim(-A*1.1, A*1.1)
ax2.set_title('Position vs Time')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Position (m)')
line_position, = ax2.plot([], [], '-', lw=2)

# Setup the velocity vs time plot in ax3
ax3.set_xlim(0, t[-1])
ax3.set_ylim(-A*omega*1.1, A*omega*1.1)  # Adjusting limits for velocity
ax3.set_title('Velocity vs Time')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Velocity (m/s)')
line_velocity, = ax3.plot([], [], 'b-', lw=2)

# Setup the acceleration vs time plot in ax4
ax4.set_xlim(0, t[-1])
ax4.set_ylim(-A*omega**2*1.1, A*omega**2*1.1)  # Adjusting limits for acceleration
ax4.set_title('Acceleration vs Time')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Acceleration (m/s^2)')
line_acceleration, = ax4.plot([], [], 'r-', lw=2)

# Animation function
def animate(i):
    # Oscillator visualization
    spring.set_data([-1, x[i]], [0, 0])
    mass.set_data([x[i], x[i]], [0, 0])  # -0.1 to simulate some displacement due to spring
    
    # Update position plot
    line_position.set_data(t[:i], x[:i])
    
    # Update velocity plot
    line_velocity.set_data(t[:i], v[:i])

    # Update acceleration plot
    line_acceleration.set_data(t[:i], ac[:i])
    
    return spring, mass, line_position, line_velocity, line_acceleration

# Initialization function for the animation
def init():
    spring.set_data([], [])
    mass.set_data([], [])
    line_position.set_data([], [])
    line_velocity.set_data([],[])
    line_acceleration.set_data([], [])
    return spring, mass, line_position, line_velocity, line_acceleration

ani = FuncAnimation(fig, animate, frames=len(t), init_func=init, interval=dt*1000, blit=True)

plt.tight_layout()
plt.show()