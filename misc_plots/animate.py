import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
plt.style.use('dark_background') 
# FILE_PATH = 'outputs_rawdata/run_4d9ed424eaa51022.h5' #dipole 
FILE_PATH = 'outputs_rawdata/run_a7b4deb3c2df7d53.h5'
INTEGRATOR = 'ps' 
DECIMATION = 1     
CUSTOM_BG = '#212330' 

# # 1. Load Data
# with h5py.File(FILE_PATH, 'r') as f:
#     data = f[f'{INTEGRATOR}/y'][:, ::DECIMATION]
#     x, y, z = data[0], data[1], data[2]

# Modified Loading Section to compare both
with h5py.File(FILE_PATH, 'r') as f:
    # Load PS data
    data_ps = f['ps/y'][:, ::DECIMATION]
    # Load RK data
    data_rk = f['rk4/y'][:, ::DECIMATION]

# Set your active plotting variables
x, y, z = data_ps[0], data_ps[1], data_ps[2]
x_rk, y_rk, z_rk = data_rk[0], data_rk[1], data_rk[2]

# --- 2D Animation Setup ---
fig2d, ax2d = plt.subplots(figsize=(8, 6))
fig2d.patch.set_facecolor(CUSTOM_BG)
ax2d.set_facecolor(CUSTOM_BG)

line2d, = ax2d.plot([], [], color='#1C9175', lw=1, alpha=0.8)
point2d, = ax2d.plot([], [], marker='o', color='#C75F09', markersize=4)

# REMOVE ALL 2D AXES
ax2d.set_axis_off()

def init_2d():
    ax2d.set_xlim(np.min(x), np.max(x))
    ax2d.set_ylim(np.min(y), np.max(y))
    return line2d, point2d

def update_2d(frame):
    line2d.set_data(x[:frame], y[:frame])
    point2d.set_data([x[frame]], [y[frame]])
    return line2d, point2d

# --- 3D Animation Setup ---
fig3d = plt.figure(figsize=(10, 8))
fig3d.patch.set_facecolor(CUSTOM_BG)
ax3d = fig3d.add_subplot(111, projection='3d')
ax3d.set_facecolor(CUSTOM_BG)

line3d, = ax3d.plot([], [], [], color='#1C9175', lw=1, alpha=0.8)
point3d, = ax3d.plot([], [], [], marker='o', color='#C75F09', markersize=4)

# REMOVE ALL 3D AXES AND GRIDS
ax3d.set_axis_off()

def init_3d():
    ax3d.set_xlim(np.min(x), np.max(x))
    ax3d.set_ylim(np.min(y), np.max(y))
    ax3d.set_zlim(np.min(z), np.max(z))
    return line3d, point3d

def update_3d(frame):
    line3d.set_data(x[:frame], y[:frame])
    line3d.set_3d_properties(z[:frame])
    point3d.set_data([x[frame]], [y[frame]])
    point3d.set_3d_properties([z[frame]])
    return line3d, point3d

# Run Animations
ani2d = FuncAnimation(fig2d, update_2d, frames=len(x), init_func=init_2d, blit=True, interval=30)
ani3d = FuncAnimation(fig3d, update_3d, frames=len(x), init_func=init_3d, blit=True, interval=30)

# --- Save Section ---
print("Saving videos...")
ani2d.save('trajectory_2d.mp4', writer='ffmpeg', fps=70, savefig_kwargs={'facecolor': CUSTOM_BG})
ani3d.save('trajectory_3d_fullgyro.mp4', writer='ffmpeg', fps=70, savefig_kwargs={'facecolor': CUSTOM_BG})

plt.show()
