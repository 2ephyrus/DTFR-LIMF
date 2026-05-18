import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# 设置字体为Times New Roman
plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.labelpad'] = 15
plt.rcParams['grid.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.3

# 数据
adaptive_reset = [0.015163, 0.016484, 0.017545, 0.020668, 0.030906, 0.027181, 0.016194, 0.002236]
hard_reset = [0.026612, 0.020505, 0.017003, 0.018165, 0.020632, 0.020617, 0.030772, 0.000808]
soft_reset = [0.015179, 0.016955, 0.017687, 0.023999, 0.021648, 0.021702, 0.034372, 0.000498]

# 计算平均值
adaptive_mean = np.mean(adaptive_reset)
hard_mean = np.mean(hard_reset)
soft_mean = np.mean(soft_reset)

# 层编号
layers = np.arange(1, len(adaptive_reset) + 1)
bar_width = 0.25

# 创建图形，使用更大的画布
fig, ax = plt.subplots(figsize=(14, 5))

# 更和谐的颜色方案 - 专业期刊常用配色
colors = ['#5DA5DA', '#FAA43A', '#60BD68']  # 蓝、橙、绿，饱和度适中

# 绘制柱状图，增加柱体边缘线增强轮廓
bars1 = ax.bar(layers - bar_width, adaptive_reset, bar_width, label='Adaptive Reset',
               color=colors[0], edgecolor='white', linewidth=1.5, alpha=0.95)
bars2 = ax.bar(layers, hard_reset, bar_width, label='Hard Reset',
               color=colors[1], edgecolor='white', linewidth=1.5, alpha=0.95)
bars3 = ax.bar(layers + bar_width, soft_reset, bar_width, label='Soft Reset',
               color=colors[2], edgecolor='white', linewidth=1.5, alpha=0.95)

# 轴标签设置
ax.set_xlabel('Layer Index', fontsize=28, fontweight='bold')
ax.set_ylabel('Firing Rate', fontsize=28, fontweight='bold')

# 刻度设置
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.set_xticks(layers)
ax.set_xticklabels(layers)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

# 更精致的网格线
ax.grid(axis='y', linestyle='-', alpha=0.3)

# 图例优化
legend = ax.legend(
    [f'Adaptive (Avg Fr: {adaptive_mean:.4f})',
     f'Hard (Avg Fr: {hard_mean:.4f})',
     f'Soft (Avg Fr: {soft_mean:.4f})'],
    fontsize=24,
    loc='upper left',
    frameon=True,
    framealpha=0.0,
    edgecolor='#e0e0e0',
    facecolor='white',
    borderaxespad=0.8,
    labelspacing=0.1
)
legend.get_frame().set_boxstyle('round,pad=0.5')

# 美化边框 - 只保留底部和左侧边框，更现代
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# 调整y轴范围，使数据展示更合理
y_min, y_max = ax.get_ylim()
ax.set_ylim(0, y_max * 1.1)

# 紧凑布局
plt.tight_layout()

# 显示图形
plt.show()
