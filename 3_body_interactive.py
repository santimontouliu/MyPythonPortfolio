import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
import tkinter as tk
from tkinter import ttk, messagebox

# Initial default values
default_masses = [1e30, 1e30, 1e30]
default_positions = [[1e11, 0], [-0.5e11, 0.866e11], [-0.5e11, -0.866e11]]
default_velocities = [[0, 10e3], [5e3, 0], [-5e3, -5e3]]

def create_input_window():
    window = tk.Tk()
    window.title("Three Body Problem Initial Conditions")
    window.geometry('400x600')
    window.configure(bg='#2E2E2E')  # Dark background

    entries = {}

    # Styling
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TLabel', background='#2E2E2E', foreground='white')
    style.configure('TEntry', fieldbackground='#404040', foreground='white')

    for i, (desc, defaults) in enumerate([("Mass", default_masses), ("Initial Position", default_positions), ("Initial Velocity", default_velocities)]):
        for j, (subdesc, default) in enumerate(zip(["Body {}".format(k+1) for k in range(3)], defaults)):
            if isinstance(default, list):
                for k, coord in enumerate(['X', 'Y']):
                    label = ttk.Label(window, text=f"{desc} {subdesc} {coord}:")
                    label.grid(row=i*6 + j*2 + k, column=0, padx=5, pady=5, sticky=tk.W)
                    entry = ttk.Entry(window)
                    entry.insert(0, str(default[k]))
                    entry.grid(row=i*6 + j*2 + k, column=1, padx=5, pady=5)
                    entries[f"{desc.lower()}_{j}_{coord.lower()}"] = entry
            else:
                label = ttk.Label(window, text=f"{desc} {subdesc}:")
                label.grid(row=i*6 + j*2, column=0, padx=5, pady=5, sticky=tk.W)
                entry = ttk.Entry(window)
                entry.insert(0, str(default))
                entry.grid(row=i*6 + j*2, column=1, padx=5, pady=5)
                entries[f"{desc.lower()}_{j}"] = entry

    def collect_inputs():
        global state, m1, m2, m3, r1, r2, r3, v1, v2, v3
        try:
            masses = [float(entries[f"mass_{i}"].get()) for i in range(3)]
            positions = [[float(entries[f"initial position_{i}_x"].get()), 
                          float(entries[f"initial position_{i}_y"].get())] for i in range(3)]
            velocities = [[float(entries[f"initial velocity_{i}_x"].get()), 
                           float(entries[f"initial velocity_{i}_y"].get())] for i in range(3)]
            
            m1, m2, m3 = masses
            r1, r2, r3 = [np.array(pos) for pos in positions]
            v1, v2, v3 = [np.array(vel) for vel in velocities]
            state = np.concatenate([r1, r2, r3, v1, v2, v3])

            window.quit()
            window.destroy()
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers for all fields.")

    ttk.Button(window, text="Start Simulation", command=collect_inputs).grid(row=18, column=0, columnspan=2, pady=20)

    window.mainloop()

    return state, m1, m2, m3, r1, r2, r3, v1, v2, v3

# Create the input window and get the values
state, m1, m2, m3, r1, r2, r3, v1, v2, v3 = create_input_window()

G = 6.67e-11 # Nm^2/kg^2

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