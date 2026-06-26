"""
Work-precision bubble plot from a hand-assembled CSV (Book2.csv).

Reads a CSV containing one or more "steps per gyro / error / wall_clock"
column blocks (one per method: RK4, PS-adaptive, PS16), located by scanning
for the 'steps per gyro' header row, and plots |ΔE|/E₀ vs wall-clock time as
a scatter where bubble size encodes steps-per-gyro. One function:

    generate_performance_plot — parse the CSV blocks and render the plot
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

def generate_performance_plot(csv_file):
    # 1. Read the file
    try:
        df_raw = pd.read_csv(csv_file, header=None)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        return

    # 2. Find the row where the data headers start
    header_idx = -1
    for i in range(len(df_raw)):
        row_str = " ".join([str(val).lower() for val in df_raw.iloc[i] if pd.notnull(val)])
        if 'steps per gyro' in row_str:
            header_idx = i
            break
            
    if header_idx == -1:
        print("Error: Could not find 'steps per gyro' in the CSV file.")
        return

    # 3. Identify column indices for the methods
    header_row = [str(val).lower().strip() if pd.notnull(val) else "" for val in df_raw.iloc[header_idx]]
    step_indices = [i for i, val in enumerate(header_row) if 'steps per gyro' in val]

    def extract_block(idx):
        if idx is None or idx >= len(header_row): return pd.DataFrame()
        block = df_raw.iloc[header_idx+1:, idx:idx+3].copy()
        block.columns = ['steps', 'error', 'wall_clock']
        return block.apply(pd.to_numeric, errors='coerce').dropna(how='any')

    rk4 = extract_block(step_indices[0] if len(step_indices) > 0 else None)
    ps_adaptive = extract_block(step_indices[1] if len(step_indices) > 1 else None)
    ps_16 = extract_block(step_indices[2] if len(step_indices) > 2 else None)

    # 4. Plotting Setup
    plt.figure(figsize=(12, 8))
    bubble_scale = 1.5 

    # Aesthetics
    methods = [
        (rk4, 'RK4', 'dodgerblue', 'o'),
        (ps_adaptive, 'PS (Adaptive)', 'darkorange', 's'),
        (ps_16, 'PS16', 'forestgreen', 'o')
    ]

    for data, label, color, marker in methods:
        if not data.empty:
            plt.scatter(data['wall_clock'], data['error'], 
                        s=data['steps'] * bubble_scale, 
                        color=color, alpha=0.6, edgecolors='none', 
                        marker=marker)
            
            for _, row in data.iterrows():
                steps = int(row['steps'])
                radius = np.sqrt(steps * bubble_scale / np.pi)
                
                # Label positioning logic
                if label == 'RK4':
                    # Labels below
                    offset = (0, -(radius + 7.5))
                    ha, va = 'center', 'top'
                elif label == 'PS (Adaptive)':
                    # Labels below - gap pushed to a "tiny bit more" (2 pts)
                    offset = (0, -(radius + 2))
                    ha, va = 'center', 'top'
                elif label == 'PS16':
                    # Labels above - gap pushed to a "tiny bit more" (1.5 pts)
                    offset = (0, radius + 1.5)
                    ha, va = 'center', 'bottom'

                plt.annotate(
                    f"{steps}", 
                    xy=(row['wall_clock'], row['error']),
                    xytext=offset,
                    textcoords="offset points",
                    ha=ha,
                    va=va,
                    fontsize=8, 
                    color=color,
                    fontweight='bold'
                )

    # 5. Formatting
    plt.yscale('log')
    plt.xlabel('Wall Clock Time (s)', fontsize=12)
    plt.ylabel(r'$|\Delta E|/E_0$', fontsize=12)
    # plt.title('Algorithm Efficiency: Error vs. Execution Time\n(Labels = Steps per Gyro Period)', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.2)
    
    # 6. Fixed-size Legend
    legend_handles = [
        mlines.Line2D([], [], color='dodgerblue', marker='o', linestyle='None', 
                      markersize=10, label='RK4', alpha=0.6),
        mlines.Line2D([], [], color='darkorange', marker='s', linestyle='None', 
                      markersize=10, label='PS (Adaptive)', alpha=0.6),
        mlines.Line2D([], [], color='forestgreen', marker='o', linestyle='None', 
                      markersize=10, label='PS (Max Order 16)', alpha=0.6)
    ]
    
    final_handles = []
    if not rk4.empty: final_handles.append(legend_handles[0])
    if not ps_adaptive.empty: final_handles.append(legend_handles[1])
    if not ps_16.empty: final_handles.append(legend_handles[2])

    plt.legend(handles=final_handles, loc='best', frameon=True)

    plt.tight_layout()
    plt.savefig('performance_plot_final.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    generate_performance_plot('Book2.csv')