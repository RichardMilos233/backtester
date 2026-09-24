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


def generate_quadrant_guide(data_path: str = 'data/SPY.csv', output_img: str = 'docs/regime_quadrant_diagram.png'):
    print(f"Loading data from {data_path}...")
    df = load_market_data(data_path)
    df_reg = classify_market_regimes(df)
    valid_df = df_reg[df_reg['regime'].notna()].copy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8.5), gridspec_kw={'width_ratios': [1.1, 1.0]})
    
    # ----------------------------------------------------
    # Plot 1: Standard Industry Conceptual 4-Quadrant Map
    # ----------------------------------------------------
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    
    # Quadrant background colors
    # Q1 (+, +): Top-Right
    ax1.add_patch(patches.Rectangle((0, 0), 1.2, 1.2, facecolor='#fee8c8', alpha=0.6, zorder=0))
    # Q2 (-, +): Top-Left
    ax1.add_patch(patches.Rectangle((-1.2, 0), 1.2, 1.2, facecolor='#fcc5c0', alpha=0.5, zorder=0))
    # Q3 (-, -): Bottom-Left
    ax1.add_patch(patches.Rectangle((-1.2, -1.2), 1.2, 1.2, facecolor='#ece7f2', alpha=0.6, zorder=0))
    # Q4 (+, -): Bottom-Right
    ax1.add_patch(patches.Rectangle((0, -1.2), 1.2, 1.2, facecolor='#e5f5e0', alpha=0.6, zorder=0))
    
    # Coordinate Axes
    ax1.axhline(0, color='#333333', lw=2.2, zorder=2)
    ax1.axvline(0, color='#333333', lw=2.2, zorder=2)
    
    # Arrows on axes
    ax1.annotate('', xy=(1.18, 0), xytext=(-1.18, 0),
                 arrowprops=dict(arrowstyle="->", color='#333333', lw=2.2))
    ax1.annotate('', xy=(0, 1.18), xytext=(0, -1.18),
                 arrowprops=dict(arrowstyle="->", color='#333333', lw=2.2))
    
    # Axis Titles & Meanings
    ax1.text(1.15, -0.08, '成交量 (Volume) → 高', ha='right', va='top', fontsize=12, fontweight='bold', color='#111111')
    ax1.text(-1.15, -0.08, '← 成交量 (Volume) 低', ha='left', va='top', fontsize=12, fontweight='bold', color='#111111')
    ax1.text(0.04, 1.15, '波动率 (Volatility) → 高', ha='left', va='top', fontsize=12, fontweight='bold', color='#111111')
    ax1.text(0.04, -1.15, '← 波动率 (Volatility) 低', ha='left', va='bottom', fontsize=12, fontweight='bold', color='#111111')
    
    # Center origin label
    ax1.text(0.03, 0.03, '原点 (0, 0)\n滚动中位数\n(Rolling Median)', fontsize=10, ha='left', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#666666', alpha=0.9))
    
    # Content of 4 Quadrants
    # Q1: Top-Right (+, +)
    ax1.text(0.6, 0.6, 
             "第一象限 Q1 (+, +)\n"
             "【高成交量 + 高波动率】\n\n"
             "• 市场特征: 恐慌暴跌、单边踩踏、巨量突破\n"
             "• 典型事件: 2020年3月疫情熔断\n"
             "• CTA 策略表现: 趋势极强，破线止损空仓\n"
             "• 策略关键词: 【危机防守 / 截断亏损】",
             ha='center', va='center', fontsize=10.5, color='#993404',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#fff7bc', edgecolor='#d95f02', lw=1.5, alpha=0.95))
    
    # Q2: Top-Left (-, +)
    ax1.text(-0.6, 0.6, 
             "第二象限 Q2 (-, +)\n"
             "【低成交量 + 高波动率】\n\n"
             "• 市场特征: 流动性枯竭真空、急涨急跌、虚假异动\n"
             "• 典型表现: 缺乏大资金承接，价格大幅跳空\n"
             "• CTA 策略表现: 极易被滑点与假突破收割\n"
             "• 策略关键词: 【流动性陷阱 / 谨慎减仓】",
             ha='center', va='center', fontsize=10.5, color='#67001f',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#fde0dd', edgecolor='#e7298a', lw=1.5, alpha=0.95))
    
    # Q3: Bottom-Left (-, -)
    ax1.text(-0.6, -0.6, 
             "第三象限 Q3 (-, -)\n"
             "【低成交量 + 低波动率】\n\n"
             "• 市场特征: 窄幅震荡、垃圾时间、方向未明\n"
             "• 典型事件: 2015-2016 磨人震荡市\n"
             "• CTA 策略表现: 频繁双向横跳被左右抽耳光 (Whipsaw)\n"
             "• 策略关键词: 【最大亏损出血点 / 建议观望】",
             ha='center', va='center', fontsize=10.5, color='#023858',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#f7fcfd', edgecolor='#7570b3', lw=1.5, alpha=0.95))
    
    # Q4: Bottom-Right (+, -)
    ax1.text(0.6, -0.6, 
             "第四象限 Q4 (+, -)\n"
             "【高成交量 + 低波动率】\n\n"
             "• 市场特征: 机构温和建仓、稳步推升、低波慢牛\n"
             "• 典型事件: 2017年单边慢牛行情\n"
             "• CTA 策略表现: 趋势极度平稳，胜率高、持仓长\n"
             "• 策略关键词: 【黄金盈利甜点区 / 重仓吃满】",
             ha='center', va='center', fontsize=10.5, color='#00441b',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#f7fcf5', edgecolor='#238b45', lw=1.5, alpha=0.95))
    
    ax1.set_title("业界标准 CTA 市场状态 4 象限理论模型 (Conceptual Model)", fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # ----------------------------------------------------
    # Plot 2: SPY Actual Historical Empirical Scatter
    # ----------------------------------------------------
    # Scatter points colored by regime
    color_map = {
        'Q1': ('#d95f02', 'Q1: 高量高波 (806天)'),
        'Q2': ('#e7298a', 'Q2: 低量高波 (414天)'),
        'Q3': ('#7570b3', 'Q3: 低量低波 (1132天)'),
        'Q4': ('#238b45', 'Q4: 高量低波 (320天)'),
    }
    
    for q_id, (col, label) in color_map.items():
        sub = valid_df[valid_df['regime'] == q_id]
        ax2.scatter(sub['rolling_volume'] / 1e6, sub['rolling_vol'] * 100,
                    c=col, s=20, alpha=0.5, label=label)
        
    # Dividing median lines for visual orientation
    med_vol = valid_df['rolling_vol'].median() * 100
    med_vlm = valid_df['rolling_volume'].median() / 1e6
    ax2.axvline(med_vlm, color='black', linestyle='--', lw=1.3, alpha=0.8)
    ax2.axhline(med_vol, color='black', linestyle='--', lw=1.3, alpha=0.8)
    
    # Annotate Q1-Q4 in actual scatter
    ax2.text(220, 75, "Q1 (+, +)", fontsize=13, fontweight='bold', color='#d95f02',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#d95f02', alpha=0.9))
    ax2.text(20, 75, "Q2 (-, +)", fontsize=13, fontweight='bold', color='#e7298a',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#e7298a', alpha=0.9))
    ax2.text(20, 8, "Q3 (-, -)", fontsize=13, fontweight='bold', color='#7570b3',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#7570b3', alpha=0.9))
    ax2.text(220, 8, "Q4 (+, -)", fontsize=13, fontweight='bold', color='#238b45',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#238b45', alpha=0.9))
    
    ax2.set_xlabel('平滑成交量 (Rolling 20D Volume, 百万股)', fontsize=11)
    ax2.set_ylabel('年化波动率 (Rolling 20D Volatility, %)', fontsize=11)
    ax2.set_title(f"标普 500 (SPY 2016-2026) 历史实证相平面分布", fontsize=13, fontweight='bold', pad=15)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='upper right', frameon=True, fontsize=10)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Quadrant diagram saved successfully to {output_img}")


if __name__ == '__main__':
    generate_quadrant_guide()
