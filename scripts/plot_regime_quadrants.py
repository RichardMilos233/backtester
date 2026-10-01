import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.data import load_market_data
from src.regimes import classify_market_regimes


def generate_quadrant_guide(ticker: str = 'SPY', output_img: str = 'docs/regime_quadrant_diagram.png'):
    print(f"Loading data for {ticker}...")
    df = load_market_data(ticker)
    df_reg = classify_market_regimes(df)
    valid_df = df_reg[df_reg['regime'].notna()].copy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8.5), gridspec_kw={'width_ratios': [1.1, 1.0]}, layout='constrained')
    
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
    
    ax1.text(1.15, -0.08, 'Volume → high', ha='right', va='top', fontsize=12, fontweight='bold', color='#111111')
    ax1.text(-1.15, -0.08, '← Volume low', ha='left', va='top', fontsize=12, fontweight='bold', color='#111111')
    ax1.text(0.04, 1.15, 'Volatility → high', ha='left', va='top', fontsize=12, fontweight='bold', color='#111111')
    ax1.text(0.04, -1.15, '← Volatility low', ha='left', va='bottom', fontsize=12, fontweight='bold', color='#111111')

    ax1.text(0.03, 0.03, 'Origin (0, 0)\nRolling median', fontsize=10, ha='left', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#666666', alpha=0.9))

    ax1.text(0.6, 0.6,
             "Q1 (+, +)\n"
             "High volume, high volatility\n\n"
             "Panic selling, liquidation,\n"
             "high-volume breakouts\n"
             "Example: March 2020\n"
             "CTA: strong trend; flatten on a break\n"
             "Role: crisis defense",
             ha='center', va='center', fontsize=10, color='#993404',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#fff7bc', edgecolor='#d95f02', lw=1.5, alpha=0.95))

    ax1.text(-0.6, 0.6,
             "Q2 (-, +)\n"
             "Low volume, high volatility\n\n"
             "Thin liquidity, gaps,\n"
             "false breakouts\n"
             "Little size behind the move\n"
             "CTA: slippage and fake breaks\n"
             "Role: liquidity trap",
             ha='center', va='center', fontsize=10, color='#67001f',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#fde0dd', edgecolor='#e7298a', lw=1.5, alpha=0.95))

    ax1.text(-0.6, -0.6,
             "Q3 (-, -)\n"
             "Low volume, low volatility\n\n"
             "Narrow range, no direction\n"
             "Example: 2015-2016 chop\n"
             "CTA: repeated whipsaws\n"
             "Role: stay out",
             ha='center', va='center', fontsize=10, color='#023858',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#f7fcfd', edgecolor='#7570b3', lw=1.5, alpha=0.95))

    ax1.text(0.6, -0.6,
             "Q4 (+, -)\n"
             "High volume, low volatility\n\n"
             "Steady accumulation,\n"
             "slow grind higher\n"
             "Example: 2017 bull market\n"
             "CTA: smooth trend, long holds\n"
             "Role: the profitable regime",
             ha='center', va='center', fontsize=10, color='#00441b',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#f7fcf5', edgecolor='#238b45', lw=1.5, alpha=0.95))

    ax1.set_title("CTA market-regime quadrants", fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # ----------------------------------------------------
    # Plot 2: SPY Actual Historical Empirical Scatter
    # ----------------------------------------------------
    # Scatter points colored by regime
    color_map = {
        'Q1': ('#d95f02', 'high volume, high vol'),
        'Q2': ('#e7298a', 'low volume, high vol'),
        'Q3': ('#7570b3', 'low volume, low vol'),
        'Q4': ('#238b45', 'high volume, low vol'),
    }

    for q_id, (col, name) in color_map.items():
        sub = valid_df[valid_df['regime'] == q_id]
        ax2.scatter(
            sub['rolling_volume'] / 1e6,
            sub['rolling_vol'] * 100,
            c=col,
            s=20,
            alpha=0.5,
            label=f'{q_id}: {name} ({len(sub)} days)',
        )

    med_vol = valid_df['rolling_vol'].median() * 100
    med_vlm = valid_df['rolling_volume'].median() / 1e6
    ax2.axvline(med_vlm, color='black', linestyle='--', lw=1.3, alpha=0.8)
    ax2.axhline(med_vol, color='black', linestyle='--', lw=1.3, alpha=0.8)

    for text, x, y, ha, va, color in [
        ("Q1 (+, +)", 0.98, 0.96, 'right', 'top', '#d95f02'),
        ("Q2 (-, +)", 0.02, 0.96, 'left', 'top', '#e7298a'),
        ("Q3 (-, -)", 0.02, 0.04, 'left', 'bottom', '#7570b3'),
        ("Q4 (+, -)", 0.98, 0.04, 'right', 'bottom', '#238b45'),
    ]:
        ax2.text(x, y, text, transform=ax2.transAxes, ha=ha, va=va, fontsize=12, fontweight='bold', color=color,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.9))
    
    ax2.set_xlabel('20-day average volume (million shares)', fontsize=11)
    ax2.set_ylabel('20-day annualized volatility (%)', fontsize=11)
    start = valid_df.index.min().year
    end = valid_df.index.max().year
    ax2.set_title(f'{ticker} regimes, {start}-{end}', fontsize=13, fontweight='bold', pad=15)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='center right', frameon=True, fontsize=8)
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Quadrant diagram saved successfully to {output_img}")


if __name__ == '__main__':
    generate_quadrant_guide()
