# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from mpl_toolkits.mplot3d import Axes3D

# # --- Configuration ---
# plt.style.use('dark_background') 
# FILE_PATH = 'outputs_rawdata/run_d9853429c7d50d25.h5'
# INTEGRATOR = 'ps' 
# DECIMATION = 5     
# CUSTOM_BG = '#212330' 

# # 1. Load Data
# with h5py.File(FILE_PATH, 'r') as f:
#     data = f[f'{INTEGRATOR}/y'][:, ::DECIMATION]
#     x, y, z = data[0], data[1], data[2]

# # --- 2D Animation Setup ---
# fig2d, ax2d = plt.subplots(figsize=(8, 6))
# fig2d.patch.set_facecolor(CUSTOM_BG)
# ax2d.set_facecolor(CUSTOM_BG)

# line2d, = ax2d.plot([], [], color='#1C9175', lw=1, alpha=0.8)
# point2d, = ax2d.plot([], [], marker='o', color='#C75F09', markersize=4)

# # REMOVE ALL 2D AXES
# ax2d.set_axis_off()

# def init_2d():
#     ax2d.set_xlim(np.min(x), np.max(x))
#     ax2d.set_ylim(np.min(y), np.max(y))
#     return line2d, point2d

# def update_2d(frame):
#     line2d.set_data(x[:frame], y[:frame])
#     point2d.set_data([x[frame]], [y[frame]])
#     return line2d, point2d

# # --- 3D Animation Setup ---
# fig3d = plt.figure(figsize=(10, 8))
# fig3d.patch.set_facecolor(CUSTOM_BG)
# ax3d = fig3d.add_subplot(111, projection='3d')
# ax3d.set_facecolor(CUSTOM_BG)

# line3d, = ax3d.plot([], [], [], color='#1C9175', lw=1, alpha=0.8)
# point3d, = ax3d.plot([], [], [], marker='o', color='#C75F09', markersize=4)

# # REMOVE ALL 3D AXES AND GRIDS
# ax3d.set_axis_off()

# def init_3d():
#     ax3d.set_xlim(np.min(x), np.max(x))
#     ax3d.set_ylim(np.min(y), np.max(y))
#     ax3d.set_zlim(np.min(z), np.max(z))
#     return line3d, point3d

# def update_3d(frame):
#     line3d.set_data(x[:frame], y[:frame])
#     line3d.set_3d_properties(z[:frame])
#     point3d.set_data([x[frame]], [y[frame]])
#     point3d.set_3d_properties([z[frame]])
#     return line3d, point3d

# # Run Animations
# ani2d = FuncAnimation(fig2d, update_2d, frames=len(x), init_func=init_2d, blit=True, interval=30)
# ani3d = FuncAnimation(fig3d, update_3d, frames=len(x), init_func=init_3d, blit=True, interval=30)

# # --- Save Section ---
# print("Saving videos...")
# ani2d.save('trajectory_2d.mp4', writer='ffmpeg', fps=30, savefig_kwargs={'facecolor': CUSTOM_BG})
# ani3d.save('trajectory_3d.mp4', writer='ffmpeg', fps=30, savefig_kwargs={'facecolor': CUSTOM_BG})

# plt.show()

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
plt.style.use('dark_background') 
FILE_PATH = 'outputs_rawdata/run_d9853429c7d50d25.h5' #dipole 
# FILE_PATH = 'outputs_rawdata/run_a7869e87a1162d78.h5'
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
    
    # Optional: Rotate 3D camera for a dynamic effect
    # ax3d.view_init(elev=20, azim=frame * 0.5)
    
    return line2d, point2d, line3d, point3d

# Run the single combined animation
ani = FuncAnimation(fig, update, frames=len(x), init_func=init, blit=True, interval=30)

# --- Save Section ---
# To save as one file:
ani.save('trajectory_combined.mp4', writer='ffmpeg', fps=50, savefig_kwargs={'facecolor': CUSTOM_BG})

plt.show()