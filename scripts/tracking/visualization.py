"""Trajectory visualization helpers for one or three camera views."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for script execution.
from matplotlib.patches import Rectangle
import seaborn as sns

# Prefer fonts that can handle a broad range of labels.
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_trajectories_single_view(df, player_name, title_suffix="", save_path=None):
    """
    Plot trajectories for a single view.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Collect fish IDs present in this view.
    fish_ids = sorted([fid for fid in df['fish_id'].unique() if fid != 'unassigned'])
    
    # Build a color map for the fish IDs.
    colors = plt.cm.tab10(np.linspace(0, 1, len(fish_ids)))
    color_map = {fid: colors[i] for i, fid in enumerate(fish_ids)}
    
    # Estimate the image height so the y-axis can be flipped into a
    # standard mathematical coordinate system.
    H = None
    if 'y' in df.columns:
        y_valid = df['y'].dropna().values
        if y_valid.size > 0:
            H = float(np.max(y_valid))

    # Plot one trajectory per fish.
    for fish_id in fish_ids:
        fish_data = df[df['fish_id'] == fish_id].sort_values('frame')
        
        if len(fish_data) == 0:
            continue
        
        # Ignore rows with missing coordinates.
        valid_data = fish_data.dropna(subset=['x', 'y'])
        
        if len(valid_data) == 0:
            continue
        
        x = valid_data['x'].values
        y_img = valid_data['y'].values

        if H is not None:
            y = H - y_img
        else:
            y = y_img

        # Plot the full trajectory line.
        ax.plot(x, y, '-', color=color_map[fish_id], alpha=0.6, linewidth=1.5, label=f'Fish {fish_id}')

        # Mark the start and end points.
        if len(x) > 0:
            ax.plot(x[0], y[0], 'o', color=color_map[fish_id], markersize=8, markeredgecolor='black', markeredgewidth=1)
            ax.plot(x[-1], y[-1], 's', color=color_map[fish_id], markersize=8, markeredgecolor='black', markeredgewidth=1)
    
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.set_title(f'{player_name} - {title_suffix}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    # Make the y-axis increase upward.
    if H is not None:
        ax.set_ylim(0, H)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return None, None
    
    return fig, ax


def plot_trajectories_three_views(df_dict, title_suffix="", save_path=None):
    """
    Plot trajectories for all three views on one figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    players = ['player1', 'player2', 'player3']
    player_names = {
        'player1': 'Top View (PLAYER1)',
        'player2': 'Side View 1 (PLAYER2)',
        'player3': 'Side View 2 (PLAYER3)'
    }
    
    # Collect all fish IDs across views so colors stay consistent.
    all_fish_ids = set()
    for df in df_dict.values():
        all_fish_ids.update([fid for fid in df['fish_id'].unique() if fid != 'unassigned'])
    all_fish_ids = sorted(list(all_fish_ids))
    
    # Shared color map across views.
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_fish_ids)))
    color_map = {fid: colors[i] for i, fid in enumerate(all_fish_ids)}
    
    for idx, player in enumerate(players):
        ax = axes[idx]
        df = df_dict.get(player)
        
        if df is None or len(df) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(player_names.get(player, player), fontsize=12)
            continue
        
        # Fish IDs present in this view.
        fish_ids = sorted([fid for fid in df['fish_id'].unique() if fid != 'unassigned'])
        
        # Estimate the image height for y-axis flipping.
        H = None
        if 'y' in df.columns:
            y_valid = df['y'].dropna().values
            if y_valid.size > 0:
                H = float(np.max(y_valid))

        # Plot one trajectory per fish.
        for fish_id in fish_ids:
            fish_data = df[df['fish_id'] == fish_id].sort_values('frame')
            
            if len(fish_data) == 0:
                continue
            
            # Ignore rows with missing coordinates.
            valid_data = fish_data.dropna(subset=['x', 'y'])
            
            if len(valid_data) == 0:
                continue
            
            x = valid_data['x'].values
            y_img = valid_data['y'].values

            if H is not None:
                y = H - y_img
            else:
                y = y_img

            # Plot the trajectory line.
            ax.plot(x, y, '-', color=color_map[fish_id], alpha=0.7, linewidth=2, label=f'Fish {fish_id}')

            # Mark start and end points.
            if len(x) > 0:
                ax.plot(x[0], y[0], 'o', color=color_map[fish_id], markersize=10,
                       markeredgecolor='black', markeredgewidth=1.5, label='Start' if fish_id == fish_ids[0] else '')
                ax.plot(x[-1], y[-1], 's', color=color_map[fish_id], markersize=10,
                       markeredgecolor='black', markeredgewidth=1.5, label='End' if fish_id == fish_ids[0] else '')
        
        ax.set_xlabel('X (pixels)', fontsize=11)
        ax.set_ylabel('Y (pixels)', fontsize=11)
        ax.set_title(player_names.get(player, player), fontsize=12, fontweight='bold')
        if idx == 0:
            ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        # Make the y-axis increase upward.
        if H is not None:
            ax.set_ylim(0, H)
    
    plt.suptitle(f'Trajectories - {title_suffix}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return None
    
    return fig
