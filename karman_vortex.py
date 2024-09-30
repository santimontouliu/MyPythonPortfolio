import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
NUM_PANELS = 50
CYLINDER_RADIUS = 0.5
FLOW_VELOCITY = 1.0
VORTEX_STRENGTH = 0.5
NUM_VORTICES = 40
FPS = 30
FRAMES = 300

# Generate points on the cylinder
theta = np.linspace(0, 2*np.pi, NUM_PANELS)
cylinder_x = CYLINDER_RADIUS * np.cos(theta)
cylinder_y = CYLINDER_RADIUS * np.sin(theta)

# Initial setup
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(-2, 10)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
streamline, = ax.plot([], [], 'b-', lw=0.5, alpha=0.5)  # Lighter and thinner streamlines
ax.plot(cylinder_x, cylinder_y, 'r-', lw=2)  # Draw the cylinder

# Text for explanation placed outside the plot area
text = fig.text(0.01, 0.8, '', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=5))

# Initialize vortices outside the plot to avoid initial overlap with the cylinder
vortex_x = np.random.uniform(-2, -1, NUM_VORTICES)
vortex_y = np.random.uniform(-3, 3, NUM_VORTICES)
vortex_gamma = np.random.choice([-1, 1], NUM_VORTICES) * VORTEX_STRENGTH

# Panel method to solve for vortex sheet strength on cylinder
def solve_panel_method():
    A = np.zeros((NUM_PANELS, NUM_PANELS))
    for i in range(NUM_PANELS):
        for j in range(NUM_PANELS):
            if i == j:
                A[i, j] = 0
            else:
                dx = cylinder_x[i] - cylinder_x[j]
                dy = cylinder_y[i] - cylinder_y[j]
                r = np.hypot(dx, dy)
                A[i, j] = -dx / (2 * np.pi * r**2)
                
    b = -FLOW_VELOCITY * np.sin(theta)
    vortex_sheet_strength = np.linalg.solve(A, b)
    return vortex_sheet_strength

vortex_sheet_strength = solve_panel_method()

def velocity_at_point(px, py):
    u = FLOW_VELOCITY
    v = 0
    for i in range(NUM_PANELS):
        r = np.hypot(px - cylinder_x[i], py - cylinder_y[i])
        if r < 1e-5: continue  # Avoid singularity at the panel itself
        theta_panel = np.arctan2(py - cylinder_y[i], px - cylinder_x[i])
        u_panel = (vortex_sheet_strength[i] / (2 * np.pi * r)) * (py - cylinder_y[i])
        v_panel = -(vortex_sheet_strength[i] / (2 * np.pi * r)) * (px - cylinder_x[i])
        u += u_panel
        v += v_panel
    
    # Add influence from free vortices
    for vx, vy, gamma in zip(vortex_x, vortex_y, vortex_gamma):
        r_sq = (px - vx)**2 + (py - vy)**2
        if r_sq < 1e-5: continue  # Skip if too close to vortex center
        u -= gamma * (py - vy) / (2 * np.pi * r_sq)
        v += gamma * (px - vx) / (2 * np.pi * r_sq)
    
    return u, v

def animate(frame):
    global vortex_x, vortex_y, vortex_gamma
    
    for i in range(len(vortex_x)):
        u, v = velocity_at_point(vortex_x[i], vortex_y[i])
        vortex_x[i] += u * (1/FPS)
        vortex_y[i] += v * (1/FPS)
        
        if vortex_x[i] > 10 or abs(vortex_y[i]) > 3 or abs(vortex_gamma[i]) < 0.01:
            vortex_x[i], vortex_y[i], vortex_gamma[i] = np.random.uniform(-2, -1), np.random.uniform(-3, 3), np.random.choice([-1, 1]) * VORTEX_STRENGTH

    if frame % 2 == 0:  # Reduce streamline density further
        x_start = np.random.uniform(-2, 10, 1)
        y_start = np.random.uniform(-3, 3, 1)
        x_vals, y_vals = [x_start[0]], [y_start[0]]
        for _ in range(50):  # Shorter streamlines for less clutter
            u, v = velocity_at_point(x_vals[-1], y_vals[-1])
            if abs(u) > 10 or abs(v) > 10: break  # Stop if velocity becomes unrealistic
            x_vals.append(x_vals[-1] + u * 0.05)
            y_vals.append(y_vals[-1] + v * 0.05)
        ax.plot(x_vals, y_vals, 'b-', lw=0.5, alpha=0.5)
    
    explanation = [
        "Kármán Vortex Street Simulation",
        f"Frame: {frame}/{FRAMES}",
        "Vortices create pressure differences, leading to alternating flow pattern.",
        "Red circle: the cylinder.",
        "Blue lines: fluid flow streamlines."
    ]
    text.set_text("\n".join(explanation))

    return [streamline, text]

fig.subplots_adjust(left=0.05, right=0.75)  # Adjust plot area to make room for text
anim = FuncAnimation(fig, animate, frames=FRAMES, interval=1000/FPS, blit=False)
plt.title('Kármán Vortex Street Simulation', y=1.08)
plt.show()