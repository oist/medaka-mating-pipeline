"""
可视化功能：显示轨迹
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from matplotlib.patches import Rectangle
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_trajectories_single_view(df, player_name, title_suffix="", save_path=None):
    """
    绘制单个视角的轨迹
    
    Args:
        df: DataFrame with columns ['frame', 'fish_id', 'x', 'y']
        player_name: 视角名称
        title_suffix: 标题后缀
        save_path: 保存路径（如果为None则显示）
        
    Returns:
        fig, ax: matplotlib图形对象
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 获取所有fish_id
    fish_ids = sorted([fid for fid in df['fish_id'].unique() if fid != 'unassigned'])
    
    # 颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, len(fish_ids)))
    color_map = {fid: colors[i] for i, fid in enumerate(fish_ids)}
    
    # 计算该视角的图像高度 H，用于把图像坐标系 (0 在上) 翻到数学坐标系 (0 在下)
    H = None
    if 'y' in df.columns:
        y_valid = df['y'].dropna().values
        if y_valid.size > 0:
            H = float(np.max(y_valid))

    # 绘制每条鱼的轨迹（在画之前把 y 做变换：y_plot = H - y_img）
    for fish_id in fish_ids:
        fish_data = df[df['fish_id'] == fish_id].sort_values('frame')
        
        if len(fish_data) == 0:
            continue
        
        # 过滤NaN值
        valid_data = fish_data.dropna(subset=['x', 'y'])
        
        if len(valid_data) == 0:
            continue
        
        x = valid_data['x'].values
        y_img = valid_data['y'].values

        if H is not None:
            y = H - y_img
        else:
            y = y_img

        # 绘制轨迹线
        ax.plot(x, y, '-', color=color_map[fish_id], alpha=0.6, linewidth=1.5, label=f'Fish {fish_id}')

        # 标记起点和终点
        if len(x) > 0:
            ax.plot(x[0], y[0], 'o', color=color_map[fish_id], markersize=8, markeredgecolor='black', markeredgewidth=1)
            ax.plot(x[-1], y[-1], 's', color=color_map[fish_id], markersize=8, markeredgecolor='black', markeredgewidth=1)
    
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.set_title(f'{player_name} - {title_suffix}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    # 设定 y 轴为数学坐标系：左下角为 (0,0)，y 向上增大
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
    绘制三个视角的轨迹（一张大图，3个小图）
    
    Args:
        df_dict: {player: DataFrame}
        title_suffix: 标题后缀
        save_path: 保存路径
        
    Returns:
        fig: matplotlib图形对象
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    players = ['player1', 'player2', 'player3']
    player_names = {
        'player1': 'Top View (PLAYER1)',
        'player2': 'Side View 1 (PLAYER2)',
        'player3': 'Side View 2 (PLAYER3)'
    }
    
    # 获取所有fish_id（用于统一颜色映射）
    all_fish_ids = set()
    for df in df_dict.values():
        all_fish_ids.update([fid for fid in df['fish_id'].unique() if fid != 'unassigned'])
    all_fish_ids = sorted(list(all_fish_ids))
    
    # 统一颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_fish_ids)))
    color_map = {fid: colors[i] for i, fid in enumerate(all_fish_ids)}
    
    for idx, player in enumerate(players):
        ax = axes[idx]
        df = df_dict.get(player)
        
        if df is None or len(df) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(player_names.get(player, player), fontsize=12)
            continue
        
        # 获取该视角的fish_id
        fish_ids = sorted([fid for fid in df['fish_id'].unique() if fid != 'unassigned'])
        
        # 为该视角计算统一的图像高度 H，用于 y 轴变换
        H = None
        if 'y' in df.columns:
            y_valid = df['y'].dropna().values
            if y_valid.size > 0:
                H = float(np.max(y_valid))

        # 绘制每条鱼的轨迹（y_plot = H - y_img）
        for fish_id in fish_ids:
            fish_data = df[df['fish_id'] == fish_id].sort_values('frame')
            
            if len(fish_data) == 0:
                continue
            
            # 过滤NaN值
            valid_data = fish_data.dropna(subset=['x', 'y'])
            
            if len(valid_data) == 0:
                continue
            
            x = valid_data['x'].values
            y_img = valid_data['y'].values

            if H is not None:
                y = H - y_img
            else:
                y = y_img

            # 绘制轨迹线
            ax.plot(x, y, '-', color=color_map[fish_id], alpha=0.7, linewidth=2, label=f'Fish {fish_id}')

            # 标记起点和终点
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
        # 设定 y 轴为数学坐标系：左下角为 (0,0)，y 向上增大
        if H is not None:
            ax.set_ylim(0, H)
    
    plt.suptitle(f'Trajectories - {title_suffix}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return None
    
    return fig
