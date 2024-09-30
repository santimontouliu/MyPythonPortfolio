import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Setup grid
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
water_height = np.zeros(X.shape)

# Initial wave disturbance
water_height[45:55, 45:55] = 1  # A square wave source in the middle

fig, ax = plt.subplots()
contour = ax.imshow(water_height, cmap='ocean', extent=[-10, 10, -10, 10], origin='lower')

def animate(frame):
    global water_height
    
    # Very simplistic wave equation simulation
    d2x = np.roll(water_height, 1, axis=1) + np.roll(water_height, -1, axis=1) - 2 * water_height
    d2y = np.roll(water_height, 1, axis=0) + np.roll(water_height, -1, axis=0) - 2 * water_height
    water_height += 0.01 * (d2x + d2y)  # Time step and wave speed are simplified here
    
    # Update the plot
    contour.set_array(water_height)
    return [contour]

anim = FuncAnimation(fig, animate, frames=500, interval=10, blit=True)
plt.title('Wave Propagation')
plt.show()