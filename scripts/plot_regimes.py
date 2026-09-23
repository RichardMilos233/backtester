import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np

# Configure Chinese font support for macOS
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Heiti SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

from src.data import load_market_data
from src.regimes import classify_market_regimes


def plot_market_regimes(data_path: str = 'data/SPY.csv', output_img: str = 'docs/market_regimes.png'):
    print(f"Loading data from {data_path}...")
    df = load_market_data(data_path)
    df_reg = classify_market_regimes(df)
    
    # Filter to valid regime period
    valid_df = df_reg[df_reg['regime'].notna()].copy()
    
    # Color palette and readable labels for each quadrant
    regime_info = {
        'Q1': {'name': 'Q1: 高量高波 (High Vol, High Volume)', 'color': '#d95f02', 'desc': '危机恐慌 / 剧烈放量突破'},
        'Q2': {'name': 'Q2: 低量高波 (High Vol, Low Volume)',  'color': '#7570b3', 'desc': '流动性真空 / 缩量剧震'},
        'Q3': {'name': 'Q3: 低量低波 (Low Vol, Low Volume)',   'color': '#80b1d3', 'desc': '窄幅横盘 / 垃圾时间'},
        'Q4': {'name': 'Q4: 高量低波 (Low Vol, High Volume)',  'color': '#2ca02c', 'desc': '机构建仓 / 温和慢牛'},
    }
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1.0], hspace=0.3)
    
    # -------------------------------------------------------------
    # Panel 1: Cartesian 2D Coordinate System (Phase Plane)
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0])
    
    # Scatter plot of all valid days
    for q_id, info in regime_info.items():
        q_data = valid_df[valid_df['regime'] == q_id]
        ax1.scatter(
            q_data['rolling_volume'] / 1e6, 
            q_data['rolling_vol'] * 100, 
            c=info['color'], 
            alpha=0.45, 
            s=22, 
            label=f"{info['name']} ({len(q_data)}天)"
        )
    
    # Dividing median reference lines (overall median for visual boundary)
    med_vol = valid_df['rolling_vol'].median() * 100
    med_vlm = valid_df['rolling_volume'].median() / 1e6
    
    ax1.axvline(med_vlm, color='black', linestyle='--', lw=1.2, alpha=0.7)
    ax1.axhline(med_vol, color='black', linestyle='--', lw=1.2, alpha=0.7)
    
    # Quadrant Callout Text in 4 corners
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    
    ax1.text(xlim[1]*0.98, ylim[1]*0.95, "第一象限 Q1 (+, +)\n高量 + 高波", 
             ha='right', va='top', fontsize=11, fontweight='bold', color=regime_info['Q1']['color'],
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=regime_info['Q1']['color'], alpha=0.9))
    
    ax1.text(xlim[0]*1.02, ylim[1]*0.95, "第二象限 Q2 (-, +)\n低量 + 高波", 
             ha='left', va='top', fontsize=11, fontweight='bold', color=regime_info['Q2']['color'],
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=regime_info['Q2']['color'], alpha=0.9))
    
    ax1.text(xlim[0]*1.02, ylim[0]*1.05, "第三象限 Q3 (-, -)\n低量 + 低波", 
             ha='left', va='bottom', fontsize=11, fontweight='bold', color=regime_info['Q3']['color'],
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=regime_info['Q3']['color'], alpha=0.9))
    
    ax1.text(xlim[1]*0.98, ylim[0]*1.05, "第四象限 Q4 (+, -)\n高量 + 低波", 
             ha='right', va='bottom', fontsize=11, fontweight='bold', color=regime_info['Q4']['color'],
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=regime_info['Q4']['color'], alpha=0.9))
    
    ax1.set_title('数学笛卡尔坐标系: 市场状态 4 象限分布 (Volume vs Volatility Phase Plane)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('平滑成交量 (Rolling 20D Volume, 百万股 / 百万)', fontsize=11)
    ax1.set_ylabel('年化波动率 (Rolling 20D Volatility, %)', fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True, fontsize=10)
    
    # -------------------------------------------------------------
    # Panel 2: SPY Price Time Series with Regime Background Bands
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(valid_df.index, valid_df['close'], color='#222222', lw=1.5, label='SPY 收盘价')
    
    # Draw regime colored background spans
    for q_id, info in regime_info.items():
        q_mask = valid_df['regime'] == q_id
        # Group contiguous days to draw span
        changes = (q_mask != q_mask.shift(1))
        group_ids = changes.cumsum()
        for _, grp in valid_df[q_mask].groupby(group_ids):
            start_date = grp.index[0]
            end_date = grp.index[-1]
            ax2.axvspan(start_date, end_date, color=info['color'], alpha=0.22)
            
    # Add dummy patches for legend in ax2
    legend_patches = [
        patches.Patch(facecolor=info['color'], alpha=0.4, label=f"{q_id}: {info['desc']}")
        for q_id, info in regime_info.items()
    ]
    legend_patches.insert(0, plt.Line2D([0], [0], color='#222222', lw=1.5, label='SPY Price'))
    ax2.legend(handles=legend_patches, loc='upper left', fontsize=9, frameon=True)
    
    ax2.set_title('SPY 历史走势与市场状态 (Regimes) 时序变迁', fontsize=13, fontweight='bold')
    ax2.set_xlabel('日期', fontsize=11)
    ax2.set_ylabel('SPY 价格 ($)', fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved successfully to {output_img}")


if __name__ == '__main__':
    plot_market_regimes()
