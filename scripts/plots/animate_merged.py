"""
Side-by-side 2-D (x-y) and 3-D trajectory video of one solver's orbit.

No CLI: edit FILE_PATH / INTEGRATOR / DECIMATION below.
Input:  an h5 run with <INTEGRATOR>/y rows [x, y, z, ...].
Output: trajectory_combined.mp4 in the cwd, 50 fps, dark background, one frame
        per DECIMATION-th sample. Keep samples/DECIMATION to a few thousand:
        that is the frame count, and the decimated orbit is held in memory.
Needs ffmpeg on PATH (checked before rendering).
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation

# --- Configuration ---
plt.style.use('dark_background') 
FILE_PATH = 'outputs_rawdata/run_d9853429c7d50d25.h5' #dipole 
INTEGRATOR = 'ps' 
DECIMATION = 5     #5 for dipole
CUSTOM_BG = '#212330' 

# 1. Load Data
with h5py.File(FILE_PATH, 'r') as f:
    data = f[f'{INTEGRATOR}/y'][:, ::DECIMATION]
    x, y, z = data[0], data[1], data[2]

# --- Combined Figure Setup ---
# Create a figure with two subplots: one 2D and one 3D
fig = plt.figure(figsize=(16, 8))
fig.patch.set_facecolor(CUSTOM_BG)

# 2D Subplot (Left)
ax2d = fig.add_subplot(121)
ax2d.set_facecolor(CUSTOM_BG)
line2d, = ax2d.plot([], [], color='#1C9175', lw=1, alpha=0.8)
point2d, = ax2d.plot([], [], marker='o', color='#C75F09', markersize=4)
ax2d.set_axis_off()

# 3D Subplot (Right)
ax3d = fig.add_subplot(122, projection='3d')
ax3d.set_facecolor(CUSTOM_BG)
line3d, = ax3d.plot([], [], [], color='#1C9175', lw=1, alpha=0.8)
point3d, = ax3d.plot([], [], [], marker='o', color='#C75F09', markersize=4)
ax3d.set_axis_off()

def init():
    # Set limits for 2D
    ax2d.set_xlim(np.min(x), np.max(x))
    ax2d.set_ylim(np.min(y), np.max(y))
    # Set limits for 3D
    ax3d.set_xlim(np.min(x), np.max(x))
    ax3d.set_ylim(np.min(y), np.max(y))
    ax3d.set_zlim(np.min(z), np.max(z))
    return line2d, point2d, line3d, point3d

def update(frame):
    # Update 2D Data
    line2d.set_data(x[:frame], y[:frame])
    point2d.set_data([x[frame]], [y[frame]])
    
    # Update 3D Data
    line3d.set_data(x[:frame], y[:frame])
    line3d.set_3d_properties(z[:frame])
    point3d.set_data([x[frame]], [y[frame]])
    point3d.set_3d_properties([z[frame]])
    
    return line2d, point2d, line3d, point3d

# Run the single combined animation
ani = FuncAnimation(fig, update, frames=len(x), init_func=init, blit=True, interval=30)

# --- Save Section ---
if not animation.FFMpegWriter.isAvailable():
    raise SystemExit("ffmpeg not found on PATH; install it or set matplotlib's animation.ffmpeg_path")
ani.save('trajectory_combined.mp4', writer='ffmpeg', fps=50, savefig_kwargs={'facecolor': CUSTOM_BG})

plt.show()