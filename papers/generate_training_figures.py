#!/usr/bin/env python3
"""生成 16-hand 训练报告论文的图表。"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 9,
    'axes.linewidth': 0.7,
    'lines.linewidth': 1.5,
    'figure.dpi': 200,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Noto Sans CJK SC', 'Noto Serif CJK SC', 'DejaVu Sans'],
    'axes.unicode_minus': False,
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(os.path.dirname(__file__), 'training_output')
os.makedirs(OUTDIR, exist_ok=True)

with open(os.path.join(ROOT, 'reports', 'training_results.json')) as f:
    R = json.load(f)

ROUNDS    = ['cpu_smoke', 'gpu_r1', 'gpu_r2', 'gpu_r3']
LABELS    = ['CPU 烟测\n500K 步', 'GPU R1\n10M 步', 'GPU R2\n295M (旧)', 'GPU R3\n240M (修复)']
COLORS    = ['#90A4AE', '#42A5F5', '#FFA726', '#66BB6A']


def fig_round_comparison():
    """4 轮训练 best 模型对比（vs random / vs greedy / SPS / 训练时长）。"""
    fig, axes = plt.subplots(1, 4, figsize=(8.5, 2.5))

    # vs random_legal
    vals_r = [R['training_rounds'][r]['best_vs_random'] for r in ROUNDS]
    bars = axes[0].bar(range(4), vals_r, color=COLORS, width=0.6,
                       edgecolor='white', linewidth=0.5)
    axes[0].set_title('vs random_legal 胜率', fontsize=8.5, fontweight='bold')
    axes[0].set_ylim(0.7, 1.0)
    for b, v in zip(bars, vals_r):
        axes[0].text(b.get_x() + b.get_width()/2, v + 0.003,
                     f'{v*100:.1f}%', ha='center', va='bottom', fontsize=7.5)

    # vs greedy_low
    vals_g = [R['training_rounds'][r]['best_vs_greedy'] for r in ROUNDS]
    bars = axes[1].bar(range(4), vals_g, color=COLORS, width=0.6,
                       edgecolor='white', linewidth=0.5)
    axes[1].set_title('vs greedy_low 胜率', fontsize=8.5, fontweight='bold')
    axes[1].set_ylim(0.65, 0.92)
    for b, v in zip(bars, vals_g):
        axes[1].text(b.get_x() + b.get_width()/2, v + 0.003,
                     f'{v*100:.1f}%', ha='center', va='bottom', fontsize=7.5)
    # 标 Stage 7 目标线
    axes[1].axhline(0.80, color='red', linestyle='--', linewidth=0.7, alpha=0.5)
    axes[1].text(3.4, 0.805, 'Stage 7 目标', fontsize=6.5, color='red',
                 ha='right', va='bottom')

    # SPS
    sps_vals = [R['training_rounds'][r].get('sps_avg', 0) for r in ROUNDS]
    bars = axes[2].bar(range(4), sps_vals, color=COLORS, width=0.6,
                       edgecolor='white', linewidth=0.5)
    axes[2].set_title('训练吞吐量 (SPS)', fontsize=8.5, fontweight='bold')
    for b, v in zip(bars, sps_vals):
        axes[2].text(b.get_x() + b.get_width()/2, v + 50,
                     f'{v}', ha='center', va='bottom', fontsize=7.5)

    # 训练步数（对数轴）
    steps = [R['training_rounds'][r].get('total_steps_actual') or
             R['training_rounds'][r].get('total_steps') for r in ROUNDS]
    bars = axes[3].bar(range(4), steps, color=COLORS, width=0.6,
                       edgecolor='white', linewidth=0.5)
    axes[3].set_yscale('log')
    axes[3].set_title('总训练步数 (log)', fontsize=8.5, fontweight='bold')
    for b, v in zip(bars, steps):
        if v >= 1e6:
            text = f'{v/1e6:.0f}M'
        else:
            text = f'{v/1e3:.0f}K'
        axes[3].text(b.get_x() + b.get_width()/2, v * 1.4,
                     text, ha='center', va='bottom', fontsize=7.5)

    for ax in axes:
        ax.set_xticks(range(4))
        ax.set_xticklabels(LABELS, fontsize=7)
        ax.tick_params(labelsize=7)

    plt.tight_layout(pad=0.4)
    fig.savefig(f'{OUTDIR}/fig_round_comparison.pdf')
    plt.close()
    print(f'  fig_round_comparison.pdf')


def fig_best_progression():
    """vs greedy_low 最优胜率随训练步数推进（4 轮 stitched 在一个时间线上）。"""
    fig, ax = plt.subplots(figsize=(7.5, 3.2))

    # 1. CPU smoke
    cpu = R['training_rounds']['cpu_smoke']
    xs = [e['step'] for e in cpu['eval_curve']]
    ys = [e['vs_greedy'] for e in cpu['eval_curve']]
    ax.plot(xs, ys, marker='o', color=COLORS[0], label='CPU 烟测 (random 对手)',
            markersize=4, linewidth=1.3)

    # 2. GPU R1: shift x 到 CPU 终点之后
    r1_x_offset = 500_000  # CPU 末步
    r1 = R['training_rounds']['gpu_r1']
    xs_r1 = [e['step'] + r1_x_offset for e in r1['best_history']]
    ys_r1 = [e['vs_greedy'] for e in r1['best_history']]
    # 加 final eval
    xs_r1.append(r1['total_steps'] + r1_x_offset)
    ys_r1.append(r1['final_eval']['vs_greedy'])
    ax.plot(xs_r1, ys_r1, marker='s', color=COLORS[1],
            label='GPU R1: 0→10M (League snapshot_upscore=2.0)',
            markersize=4, linewidth=1.3)

    # 3. GPU R2: shift x 到 R1 终点之后（resume 但实际是新 step counter，整体偏移）
    r2_x_offset = r1_x_offset + r1['total_steps']
    r2 = R['training_rounds']['gpu_r2']
    xs_r2 = [e['step'] + r2_x_offset for e in r2['best_history']]
    ys_r2 = [e['vs_greedy'] for e in r2['best_history']]
    ax.plot(xs_r2, ys_r2, marker='^', color=COLORS[2],
            label='GPU R2: resume R1 best, 0→295M (League upscore=1.0, gap=2000)',
            markersize=5, linewidth=1.3)

    # 4. GPU R3 (resume from R2's 194M.pt in zero-sum env): offset
    r3_x_offset = r2_x_offset + 194_000_000  # R3 resume from R2's step 194M
    r3 = R['training_rounds']['gpu_r3']
    xs_r3 = [e['step'] + r3_x_offset for e in r3['best_history']]
    ys_r3 = [e['vs_greedy'] for e in r3['best_history']]
    ax.plot(xs_r3, ys_r3, marker='D', color=COLORS[3],
            label='GPU R3: resume 194M.pt @ zero-sum 修复后环境, 0→240M',
            markersize=5, linewidth=1.3)

    # 目标线
    ax.axhline(0.80, color='red', linestyle='--', linewidth=0.7, alpha=0.5)
    ax.text(0.99, 0.808, 'Stage 7 目标 80%', fontsize=7, color='red',
            transform=ax.get_yaxis_transform(), ha='right')
    ax.axhline(0.65, color='orange', linestyle=':', linewidth=0.7, alpha=0.5)
    ax.text(0.99, 0.658, 'Stage 8 交付 65%', fontsize=7, color='orange',
            transform=ax.get_yaxis_transform(), ha='right')

    ax.set_xlabel('累计训练步数（拼接，对数轴）', fontsize=9)
    ax.set_ylabel('vs greedy_low 胜率', fontsize=9)
    ax.set_xscale('log')
    ax.set_ylim(0.65, 0.90)
    ax.set_title('Best 模型 vs greedy_low 胜率随训练推进', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='lower right', framealpha=0.9)
    ax.tick_params(labelsize=7.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(pad=0.4)
    fig.savefig(f'{OUTDIR}/fig_best_progression.pdf')
    plt.close()
    print(f'  fig_best_progression.pdf')


def fig_league_pool():
    """League 池规模 + 关键超参对比。"""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    pool_sizes = []
    pool_labels = []
    for r, l in zip(ROUNDS, LABELS):
        cfg = R['training_rounds'][r].get('league_config')
        if cfg and cfg.get('enabled'):
            pool_sizes.append(cfg['actual_snapshots'])
            pool_labels.append(l)

    # 左图：池规模
    colors_in_league = COLORS[1:]  # 跳过 CPU 烟测
    bars = axes[0].bar(range(len(pool_sizes)), pool_sizes,
                       color=colors_in_league, width=0.6,
                       edgecolor='white', linewidth=0.5)
    axes[0].set_title('League 快照池实际规模', fontsize=9, fontweight='bold')
    axes[0].set_xticks(range(len(pool_sizes)))
    axes[0].set_xticklabels(pool_labels, fontsize=7.5)
    axes[0].set_ylabel('快照数量', fontsize=8)
    axes[0].tick_params(labelsize=7.5)
    for b, v in zip(bars, pool_sizes):
        axes[0].text(b.get_x() + b.get_width()/2, v + 1.5,
                     f'{v}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    # 右图：snapshot_upscore + snapshot_gap 对比
    snap_data = [
        ('R1', 2.0, 1_000_000),
        ('R2', 1.0, 2000),
        ('R3', 1.0, 2000),
    ]
    x = np.arange(3)
    w = 0.35
    upscores = [d[1] for d in snap_data]
    bars1 = axes[1].bar(x - w/2, upscores, w, color='#9C27B0', label='snapshot_upscore',
                        edgecolor='white', linewidth=0.5)
    axes[1].set_ylabel('snapshot_upscore', color='#9C27B0', fontsize=8)
    axes[1].tick_params(axis='y', labelcolor='#9C27B0', labelsize=7)

    ax2 = axes[1].twinx()
    gaps = [d[2] for d in snap_data]
    bars2 = ax2.bar(x + w/2, gaps, w, color='#FF6F00', label='snapshot_gap',
                    edgecolor='white', linewidth=0.5)
    ax2.set_yscale('log')
    ax2.set_ylabel('snapshot_gap (log)', color='#FF6F00', fontsize=8)
    ax2.tick_params(axis='y', labelcolor='#FF6F00', labelsize=7)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([d[0] for d in snap_data], fontsize=8)
    axes[1].set_title('League 超参调整', fontsize=9, fontweight='bold')

    for b, v in zip(bars1, upscores):
        axes[1].text(b.get_x() + b.get_width()/2, v + 0.05,
                     f'{v}', ha='center', va='bottom', fontsize=7)
    for b, v in zip(bars2, gaps):
        ax2.text(b.get_x() + b.get_width()/2, v * 1.4,
                 f'{v//1000}k' if v >= 1000 else str(v),
                 ha='center', va='bottom', fontsize=7)

    plt.tight_layout(pad=0.4)
    fig.savefig(f'{OUTDIR}/fig_league_pool.pdf')
    plt.close()
    print(f'  fig_league_pool.pdf')


def fig_cpu_smoke_curve():
    """CPU 烟测的训练 entropy / vs greedy 曲线（验证管线收敛）。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 2.6))

    # entropy 数据（手动从 cpu log 摘录）
    entropy_steps = [20480, 40960, 61440, 81920, 102400, 122880, 143360, 163840, 184320,
                     204800, 225280, 245760, 266240, 286720, 307200, 327680, 348160,
                     368640, 389120, 409600, 430080, 450560, 471040, 491520]
    entropy_vals = [0.846, 0.736, 0.675, 0.589, 0.613, 0.564, 0.525, 0.548, 0.526,
                    0.477, 0.515, 0.491, 0.512, 0.483, 0.479, 0.460, 0.482,
                    0.458, 0.463, 0.499, 0.457, 0.425, 0.474, 0.435]
    ax1.plot(entropy_steps, entropy_vals, color='#42A5F5', linewidth=1.5,
             marker='o', markersize=3.5)
    ax1.set_xlabel('训练步数', fontsize=9)
    ax1.set_ylabel('entropy', fontsize=9)
    ax1.set_title('策略熵衰减（CPU 烟测）', fontsize=9, fontweight='bold')
    ax1.tick_params(labelsize=7.5)
    ax1.grid(True, alpha=0.3)

    cpu = R['training_rounds']['cpu_smoke']
    xs = [e['step'] for e in cpu['eval_curve']]
    ys_r = [e['vs_random'] for e in cpu['eval_curve']]
    ys_g = [e['vs_greedy'] for e in cpu['eval_curve']]
    ax2.plot(xs, ys_r, color='#42A5F5', marker='o', markersize=4, label='vs random_legal')
    ax2.plot(xs, ys_g, color='#EF5350', marker='s', markersize=4, label='vs greedy_low')
    ax2.axhline(0.80, color='red', linestyle='--', linewidth=0.7, alpha=0.5)
    ax2.set_xlabel('训练步数', fontsize=9)
    ax2.set_ylabel('胜率', fontsize=9)
    ax2.set_title('eval 胜率轨迹（CPU 烟测）', fontsize=9, fontweight='bold')
    ax2.legend(fontsize=7.5, loc='lower right')
    ax2.tick_params(labelsize=7.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.6, 0.9)

    plt.tight_layout(pad=0.4)
    fig.savefig(f'{OUTDIR}/fig_cpu_smoke.pdf')
    plt.close()
    print(f'  fig_cpu_smoke.pdf')


if __name__ == '__main__':
    print('Generating training figures →', OUTDIR)
    fig_round_comparison()
    fig_best_progression()
    fig_league_pool()
    fig_cpu_smoke_curve()
    print('Done.')
