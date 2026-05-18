from typing import Callable

import numpy as np
import torch
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from spikingjelly import visualizing
from spikingjelly.clock_driven.neuron import LIFNode as LIFNode_sj
from spikingjelly.clock_driven.neuron import ParametricLIFNode as PLIFNode_sj
from torch import nn
import matplotlib.pyplot as plt
from modules.surrogate import Rectangle, ZIF, Clip, Clip_nograd

class ILIF(LIFNode_sj):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = Rectangle(),
                 detach_reset: bool = False, cupy_fp32_inference=False, channel=0, h=0, w=0, n=1, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)
        self.register_memory('m', 0)
        # for fig
        self.register_memory('u', 0)
        # default 1
        # self.register_memory('vth', 1)
        # PLIF
        # self.tau = nn.Parameter(torch.tensor(.0), requires_grad=True)
        # KLIF
        #
        self.a = nn.Parameter(torch.tensor(.0), requires_grad=True)
        self.c = nn.Parameter(torch.tensor(.0), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.d = nn.Parameter(torch.tensor(.0), requires_grad=True)
        self.grad_h = nn.Parameter(torch.tensor(.0), requires_grad=True)
        self.new = nn.Parameter(
            torch.zeros(1, channel, 1, 1),  # 初始化形状为(1, c, 1, 1)的零张量
            requires_grad=True  # 显式指定需要计算梯度（默认也是True，可省略）
        )
        self.sg = Clip(n=n, mode=1)
        self.sg2 = Clip_nograd(n=n)
        # self.sign = Rectangle(alpha=8)
        # self.sign = Clip4Sign()



    def forward(self, x: torch.Tensor):
        if not isinstance(self.m, torch.Tensor) or self.m.shape != x.shape:
            self.m = torch.zeros_like(x, device=x.device)
        # Leaky
        self.v = self.v * 0.5
        # fig
        self.u = self.v + torch.zeros_like(x)
        # Integrate
        self.v = self.v + x
        # Vth cal
        # vth = 1
        vth = 1 + torch.tanh(self.a) * torch.tanh(x)
        # Fire
        spike = self.sg(self.v, vth)
        # Fire sign & grad learning
        sign_nograd = torch.sign(spike)
        sign_withgrad = self.grad_h.sigmoid() * spike + (sign_nograd - self.grad_h.sigmoid() * spike).detach()
        # reset voltage leaky self.m.relu() * torch.sigmoid(self.d) + -(-self.m).relu() * (1 - torch.sigmoid(self.d))
        self.m = self.m * torch.sigmoid(self.d)
        # reset voltage integrate self.m +
        self.m = self.m + (sign_withgrad * torch.sigmoid(self.b * x) - (1 - sign_withgrad) * (torch.sigmoid(self.b * x)))
        # Ad reset
        self.v = self.v - spike * vth - sign_withgrad * self.m.sigmoid()
        return spike



class softLIF(LIFNode_sj):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = Rectangle(),
                 detach_reset: bool = False, cupy_fp32_inference=False, channel=0, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)
        # FOR FIGURE
        self.register_memory('u', 0)
        self.sg = Clip(n=1, mode=1)
        self.sg2 = Clip_nograd(n=1)
        # Vth learnable
        self.a = nn.Parameter(torch.tensor(.0), requires_grad=True)
        # PLIF
        # self.tau = nn.Parameter(torch.tensor(.0), requires_grad=True)
        # KLIF
        #

    def forward(self, x: torch.Tensor):

        # Leaky and Integrate
        self.v = self.v * 0.5
        self.u = self.v + torch.zeros_like(x)
        self.v = self.v + x
        vth = 1
        # vth learn
        # vth = 1 + torch.tanh(self.a) * torch.tanh(x)
        # Spike
        spike = self.surrogate_function(self.v - 1)
        # spike = self.sg(self.v, vth)
        # spike = self.sg2(self.v, vth)
        # reset
        self.v = self.v - spike * vth

        return spike

class hardLIF(LIFNode_sj):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = Rectangle(),
                 detach_reset: bool = False, cupy_fp32_inference=False, channel=0, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)
        # FOR FIGURE
        self.register_memory('u', 0)

        self.sg = Clip(n=1, mode=1)
        self.sg2 = Clip_nograd(n=1)
        # vth = 1
        # Vth learnable
        self.a = nn.Parameter(torch.tensor(.0), requires_grad=True)
        # PLIF
        # self.tau = nn.Parameter(torch.tensor(.0), requires_grad=True)
        # KLIF
        #

    def forward(self, x: torch.Tensor):

        # Leaky and Integrate
        self.v = self.v * 0.5
        self.u = self.v + torch.zeros_like(x)
        self.v = self.v + x
        # vth learn
        vth = 1
        # vth = 1 + torch.tanh(self.a) * torch.tanh(x)
        # Spike
        spike = self.surrogate_function(self.v - 1)
        # spike = self.sg(self.v, vth)
        # spike = self.sg2(self.v, vth)
        # reset
        self.v = self.v * (1 - spike.sign())

        return spike

class XLIF(LIFNode_sj):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = Rectangle(),
                 detach_reset: bool = False, cupy_fp32_inference=False, channel=0, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)

        # ADDED Parameters
        # for fig is 0.3 ,for train is 0.0
        self.a = nn.Parameter(torch.tensor(0.3), requires_grad=True)
        self.c = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        # FOR FIGURE
        self.register_memory('s', 0)
        self.register_memory('u', 0)
        # ADDED Adaptive Reset V
        self.register_memory('m', 0)

    def forward(self, x: torch.Tensor):

        if type(self.m) is not torch.Tensor:
            self.m = torch.zeros_like(x)
        # Leaky and Integrate
        self.v = self.v * 0.5
        self.u = self.v + torch.zeros_like(x)
        self.v = self.v + x
        # Spike
        spike = self.surrogate_function(self.v - self.v_threshold - 1 * torch.tanh(self.a) * torch.tanh(x))
        self.s = spike
        # ADD
        # reset voltage leaky
        self.m = self.m.relu() * torch.sigmoid(self.c * x) + -(-self.m).relu() * (1 - torch.sigmoid(self.c * x))
        # reset voltage integrate
        self.m += spike * torch.sigmoid(x)
        self.m -= (1 - spike) * torch.sigmoid(x)
        # Adaptive reset
        self.v = self.v - spike * (self.v_threshold + 1 * torch.tanh(self.a) * torch.tanh(x) + self.m.sigmoid())

        return spike


class ReLU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)


class BPTTNeuron(LIFNode_sj):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = .0, surrogate_function: Callable = Rectangle(),
                 detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)


class PLIFNeuron(PLIFNode_sj):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = None,
                 detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset)


if __name__ == '__main__':
# FIGURE C
#     import torch
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from matplotlib.colors import LinearSegmentedColormap
#     from PIL import Image
#     import os
#
#     colors = [(0, 0, 1, 0), (0, 0, 1, 0.5), (1, 0, 0, 1)]  # RGBA值
#     cmap_name = 'neuron_cmap'
#     neuron_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
#
#
#     T = 6
#
#     def load_and_preprocess_images(image_dir, target_size=(512, 512)):
#         """
#         从指定目录加载图像并进行预处理
#
#         参数:
#         image_dir: 包含图像文件的目录路径
#         target_size: 目标图像尺寸，默认为(32, 32)
#
#         返回:
#         处理后的图像张量，形状为(T, 3, 32, 32)
#         """
#         # 获取目录中的所有图像文件
#         image_files = sorted([f for f in os.listdir(image_dir)
#                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
#
#         # 确保有足够的图像
#         if len(image_files) < T:
#             raise ValueError(f"目录中至少需要{T}张图像，当前只有{len(image_files)}张")
#
#         # 只选取前T张图像
#         image_files = image_files[:T]
#
#         # 创建用于存储图像的张量
#         x_input = torch.zeros((T, 3, *target_size))
#
#         # 加载并处理每张图像
#         for i, file_name in enumerate(image_files):
#             file_path = os.path.join(image_dir, file_name)
#             try:
#                 # 打开图像
#                 img = Image.open(file_path)
#
#                 # 转换为RGB模式（如果是灰度图会自动转换）
#                 img = img.convert('RGB')
#
#                 # 调整图像大小
#                 img = img.resize(target_size, Image.BICUBIC)
#
#                 # 转换为numpy数组
#                 img_array = np.array(img, dtype=np.float32)
#
#                 # 归一化到[0, 1]范围
#                 img_array /= 255.0
#
#                 # 转换为PyTorch张量并调整维度顺序 (H, W, C) -> (C, H, W)
#                 img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
#
#                 # 存储到输入张量中
#                 x_input[i] = img_tensor
#
#                 print(f"成功加载并处理图像: {file_name}")
#             except Exception as e:
#                 print(f"处理图像 {file_name} 时出错: {e}")
#                 continue
#
#         return x_input
#
#
#     def visualize_results(x_input, s_list1, s_list2, s_list3):
#         # 校验输入形状
#         def check_shape(s_list, name):
#             if len(s_list.shape) != 4:
#                 raise ValueError(f"{name} shape should be (T, C, H, W), got {list(s_list.shape)}")
#             T, C, H, W = s_list.shape
#             print(f"{name} shape: (time steps={T}, channels={C}, height={H}, width={W})")
#             return T, C, H, W
#
#         # 校验形状并获取时间步数量
#         T, C, H, W = check_shape(s_list1, "AR-LIF Neuron output")
#         check_shape(s_list2, "SR-LIF Neuron output")
#         check_shape(s_list3, "HR-LIF Neuron output")
#
#         # 计算时间步平均值
#         x_avg = torch.mean(x_input, dim=0)
#         s1_avg = torch.mean(s_list1, dim=0)
#         s2_avg = torch.mean(s_list2, dim=0)
#         s3_avg = torch.mean(s_list3, dim=0)
#
#         # 布局参数：极致压缩纵向空间（再增强2倍）
#         total_cols = 1 + 1 + T + 1 + 1
#         width_ratios = [0.15, 0.4] + [0.9] * T + [0.9, 0.15]
#         height_ratios = [0.95] + [0.9] * 3  # 输入行略小，神经元输出行再压缩
#         fig = plt.figure(figsize=(1.4 * T + 3, 5.5))  # 整体高度降至5.5（原8）
#         gs = fig.add_gridspec(4, total_cols, width_ratios=width_ratios, height_ratios=height_ratios)
#         # 纵向间隔再压缩2倍（0.005 → 0.002）
#         plt.subplots_adjust(wspace=0.001, hspace=0.002)
#
#         # -------------------------- Spike颜色条 --------------------------
#         cbar_spike_ax = fig.add_subplot(gs[1:3, 0])
#         cbar_spike_ax.imshow(np.array([[0], [1]]), cmap='gray', vmin=0, vmax=1)
#         cbar_spike_ax.set_yticks([0, 1])
#         cbar_spike_ax.set_yticklabels(['0', '1'], fontsize=14)  # 微调字体以适应空间
#         cbar_spike_ax.set_xticks([])
#         cbar_spike_ax.tick_params(length=0)
#         cbar_spike_ax.set_title('Spike', fontsize=16, pad=2, y=1.03)  # 进一步上移
#
#         # -------------------------- 左侧标题列 --------------------------
#         labels = ["Input", "AR-LIF Output", "SR-LIF Output", "HR-LIF Output"]
#         for row in range(4):
#             ax = fig.add_subplot(gs[row, 1])
#             ax.text(0.5, 0.5, labels[row], ha='center', va='center',
#                     fontsize=10, weight='bold', rotation=90)  # 微调字体
#             ax.axis('off')
#             for spine in ax.spines.values():
#                 spine.set_visible(False)
#
#         # -------------------------- 输入数据 --------------------------
#         for t in range(T):
#             ax = fig.add_subplot(gs[0, 2 + t])
#             img = x_input[t].permute(1, 2, 0).numpy()
#             if img.max() > 1.0:
#                 img = img / 255.0
#             ax.imshow(img, aspect='equal')
#             ax.set_title(f'T = {t}', fontsize=12, weight='bold')  # 微调字体
#             ax.axis('off')
#             for spine in ax.spines.values():
#                 spine.set_visible(False)
#
#         # 平均输入列
#         ax_avg_input = fig.add_subplot(gs[0, 2 + T])
#         avg_img = x_avg.permute(1, 2, 0).numpy()
#         if avg_img.max() > 1.0:
#             avg_img = avg_img / 255.0
#         ax_avg_input.imshow(avg_img, aspect='equal')
#         ax_avg_input.set_title('Avg T', fontsize=12, weight='bold')  # 缩短标题节省空间
#         ax_avg_input.axis('off')
#         for spine in ax_avg_input.spines.values():
#             spine.set_visible(False)
#
#         # -------------------------- 神经元输出 --------------------------
#         im_avg = None
#         neuron_lists = [s_list1, s_list2, s_list3]
#         avg_lists = [s1_avg, s2_avg, s3_avg]
#
#         for row in range(3):
#             s_list = neuron_lists[row]
#             s_avg = avg_lists[row]
#             for t in range(T):
#                 ax = fig.add_subplot(gs[row + 1, 2 + t])
#                 spikes = s_list[t, 2, :, :].detach().numpy()
#                 ax.imshow(spikes, cmap='gray', vmin=0, vmax=1, interpolation='none', aspect='equal')
#                 ax.axis('off')
#                 for spine in ax.spines.values():
#                     spine.set_visible(False)
#             # 平均列
#             ax = fig.add_subplot(gs[row + 1, 2 + T])
#             avg_spikes = s_avg[2, :, :].detach().numpy()
#             im_avg = ax.imshow(avg_spikes, cmap='viridis', vmin=0, vmax=1, interpolation='none', aspect='equal')
#             ax.axis('off')
#             for spine in ax.spines.values():
#                 spine.set_visible(False)
#
#         # -------------------------- Fire Rate颜色条 --------------------------
#         if im_avg is not None:
#             cbar_fire_ax = fig.add_subplot(gs[1:3, -1])
#             cbar_fire = fig.colorbar(im_avg, cax=cbar_fire_ax, orientation='vertical',
#                                      ticks=[0, 0.5, 1], shrink=0.65)  # 再缩短颜色条
#             cbar_fire.ax.set_yticklabels(['0', '0.5', '1'], fontsize=16)
#             cbar_fire.ax.set_title('Fire Rate', fontsize=16, pad=4, y=1.03, x=1.2)  # 进一步上移
#             cbar_fire_ax.tick_params(axis='y', length=1)  # 缩短刻度线
#
#         # 消除所有边缘留白
#         plt.tight_layout(rect=[0, 0, 1, 1])
#         plt.show()
#
#
#     def main():
#         # 用户需要修改此路径为实际存放图像的目录
#         image_directory = "E:\Code2026\Complementary-LIF/images/"
#
#         try:
#             # 加载并预处理图像
#             x_input = load_and_preprocess_images(image_directory)
#
#             # 创建三个不同配置的XLIF神经元实例
#             clif1 = XLIF()  # 标准配置
#             clif2 = softLIF()  # 较低阈值，较高时间常数
#             clif3 = hardLIF()  # 较高阈值，较低时间常数
#
#             # 为每个神经元收集输出
#             s_list1 = []
#             s_list2 = []
#             s_list3 = []
#
#             # 模拟神经元响应
#             for t in range(T):
#                 s1 = clif1(x_input[t])
#                 s2 = clif2(x_input[t])
#                 s3 = clif3(x_input[t])
#
#                 s_list1.append(s1)
#                 s_list2.append(s2)
#                 s_list3.append(s3)
#
#             # 堆叠输出以便可视化
#             s_list1 = torch.stack(s_list1, dim=0)
#             s_list2 = torch.stack(s_list2, dim=0)
#             s_list3 = torch.stack(s_list3, dim=0)
#
#             # 可视化结果
#             visualize_results(x_input, s_list1, s_list2, s_list3)
#
#         except Exception as e:
#             print(f"程序执行出错: {e}")
#
#     main()
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import torch
#
#     # 设置字体和样式（提升清晰度相关）
#     plt.rcParams["font.family"] = ["Arial", "DejaVu Sans", "sans-serif"]
#     plt.rcParams["font.size"] = 10
#     plt.rcParams["axes.labelsize"] = 16
#     plt.rcParams["xtick.labelsize"] = 14
#     plt.rcParams["ytick.labelsize"] = 14
#     plt.rcParams["legend.fontsize"] = 12
#     plt.rcParams["axes.spines.right"] = False
#     plt.rcParams["axes.spines.top"] = False
#     plt.rcParams["lines.linewidth"] = 2.5  # 线条加粗，提升清晰度
#     plt.rcParams["scatter.marker"] = 'o'
#
#     # 专业配色方案
#     colors = {
#         'input_positive': '#2A9D8F',  # 正输入 - 深青色
#         'input_negative': '#7E22CE',  # 负输入 - 紫色
#         'membrane': '#264653',  # 膜电位主色 - 靛蓝色
#         'threshold': '#0A0A0A',  # 阈值线 - 黑色
#         'spikes': '#DC2626',  # 脉冲 - 红色
#         'grid': '#E5E5E5',  # 网格线 - 浅灰色
#     }
#
#     # 优化后的8个时间步输入模式
#     input_pattern = [3.2, 1.2, 0.6, 2.1]
#     T = len(input_pattern)  # 固定8个时间步
#     reset_duration = 0.1  # 重置过程持续时间
#
#     # 创建输入张量
#     x_input = torch.tensor(input_pattern).unsqueeze(1)
#
#     # 运行神经元仿真
#     xlif = ILIF()
#     spikes = []
#     mem_before = []  # 重置前膜电位
#     mem_after = []  # 重置后膜电位
#     mem_timeline = []  # 完整时间线数据
#
#     for t in range(T):
#         s = xlif(x_input[t])
#         u = xlif.u
#         v = xlif.v
#         spikes.append(s)
#         mem_before.append(u.item())
#         mem_after.append(v.item())
#
#         # 构建时间线
#         if s > 0:  # 有脉冲
#             mem_timeline.extend([
#                 (t, u.item()),
#                 (t + reset_duration, u.item()),  # 跳变前
#                 (t + reset_duration, v.item()),  # 跳变后（垂直）
#                 (t + 1, v.item())
#             ])
#         else:  # 无脉冲
#             mem_timeline.extend([
#                 (t, v.item()),
#                 (t + 1, v.item())
#             ])
#
#     # 转换为numpy数组
#     spikes = torch.stack(spikes, dim=0).detach().numpy()
#     mem_before = np.array(mem_before)
#     mem_after = np.array(mem_after)
#     spike_mask = spikes.flatten() > 0
#
#     # ---------------------- 第一张图：输入信号图 ----------------------
#     fig1, ax_input = plt.subplots(figsize=(5, 3), dpi=300)  # 高DPI提升清晰度
#
#     input_vals = x_input[:, 0].detach().numpy()
#     positive_idx = np.where(input_vals > 0)[0]
#     negative_idx = np.where(input_vals < 0)[0]
#     zero_idx = np.where(input_vals == 0)[0]
#
#     # 绘制正输入
#     ax_input.vlines(positive_idx, 0, input_vals[positive_idx],
#                     color=colors['input_positive'], linewidth=3, alpha=0.8)
#     ax_input.scatter(positive_idx, input_vals[positive_idx],
#                      color=colors['input_positive'], s=80, zorder=3,
#                      edgecolor='white', linewidth=0.8)
#
#     # 绘制负输入
#     ax_input.vlines(negative_idx, input_vals[negative_idx], 0,
#                     color=colors['input_negative'], linewidth=3, alpha=0.8)
#     ax_input.scatter(negative_idx, input_vals[negative_idx],
#                      color=colors['input_negative'], s=80, zorder=3,
#                      edgecolor='white', linewidth=0.8)
#
#     # 标记零输入
#     ax_input.scatter(zero_idx, np.zeros_like(zero_idx),
#                      color='gray', s=80, zorder=3, marker='o',
#                      edgecolor='white', linewidth=0.8, alpha=0.5)
#
#     # 添加输入值标注
#     for t in range(T):
#         val = input_vals[t]
#         if val > 0:
#             y_pos = val + 0.2
#         elif val < 0:
#             y_pos = val - 0.2
#         else:
#             y_pos = 0.2
#         ax_input.text(t, y_pos, f'{val:.1f}',
#                       ha='center', va='center', fontsize=12,
#                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2))
#
#     # 坐标轴设置：x=0与y=0重叠
#     ax_input.set_xlim(-0.2, T - 0.5)  # x轴从0开始
#     ax_input.set_ylim(min(input_vals) - 0.5, max(input_vals) + 0.5)
#     ax_input.spines['bottom'].set_position(('data', 0))  # x轴仍在y=0
#     ax_input.spines['left'].set_position(('data', -0.1))  # 原y轴在x=0 → 移至x=-0.1（与x轴错开）
#     ax_input.set_xticks(range(0, T))
#     ax_input.set_xticklabels([str(i) for i in range(0, T)])
#     ax_input.set_xlabel('Time Step', fontsize=16, labelpad=10)  # 调整标签位置避免重叠
#     ax_input.set_ylabel('Input', fontsize=16, labelpad=10)
#     ax_input.grid(axis='y', linestyle=':', color=colors['grid'], alpha=0.7)
#
#     plt.tight_layout()
#     plt.show()
#
#     # ---------------------- 第二张图：膜电位和脉冲图 ----------------------
#     fig2, (ax_mem, ax_spike) = plt.subplots(2, 1, figsize=(8, 5), sharex=True,
#                                             gridspec_kw={'height_ratios': [3, 1]},
#                                             dpi=300)  # 高DPI
#     fig2.subplots_adjust(hspace=0.3)
#
#     # 膜电位图
#     timeline_x = [p[0] for p in mem_timeline]
#     timeline_y = [p[1] for p in mem_timeline]
#     ax_mem.plot(timeline_x, timeline_y, color=colors['membrane'], alpha=0.9)
#
#     # 阈值线
#     ax_mem.axhline(y=1.0, color=colors['threshold'], linestyle='-.',
#                    linewidth=1.0, alpha=0.8, label=r'$V_{th} = 1.0$')
#
#     # 标记脉冲时刻
#     spike_times = np.where(spike_mask)[0] if np.any(spike_mask) else []
#     for t in spike_times:
#         ax_mem.axvline(x=t, color=colors['spikes'], linestyle=':', alpha=0.3)
#
#     # 膜电位标注（确保准确性）
#     for t in range(T):
#         if spike_mask[t]:
#             val_before = mem_before[t]
#             ax_mem.text(t + 0.05, val_before + 0.15, f'{val_before:.2f}',
#                         ha='left', va='bottom', fontsize=10,
#                         color=colors['membrane'],
#                         bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))
#         val_after = mem_after[t]
#         ax_mem.text(t + 0.5, val_after, f'{val_after:.2f}',
#                     ha='center', va='center', fontsize=10,
#                     color=colors['membrane'],
#                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))
#
#     # 膜电位轴设置：x=0与y=0重叠
#     ax_mem.set_xlim(-0.2, T)
#     all_mem_vals = np.concatenate([mem_before, mem_after])
#     ax_mem.set_ylim(min(all_mem_vals) - 0.3, max(all_mem_vals) + 0.3)
#     ax_mem.spines['bottom'].set_position(('data', 0))  # x轴仍在y=0
#     ax_mem.spines['left'].set_position(('data', -0.1))  # 原y轴在x=0 → 移至x=-0.1
#     ax_mem.set_xticks(range(0, T + 1))
#     ax_mem.set_xticklabels([str(i) for i in range(0, T + 1)])
#     ax_mem.set_ylabel('Potential', fontsize=16, labelpad=10)
#     ax_mem.legend(loc='upper right', frameon=True, framealpha=0.9,
#                   edgecolor=colors['grid'], fontsize=12)
#     ax_mem.grid(True, linestyle=':', color=colors['grid'], alpha=0.7)
#
#     # 脉冲图（紧凑布局+准确显示）
#     if np.any(spike_mask):
#         spike_values = spikes[spike_mask, 0]
#         spike_idx = np.where(spike_mask)[0]
#         ax_spike.vlines(spike_idx, 0, spike_values,
#                         color=colors['spikes'], linewidth=3, alpha=0.8)
#         ax_spike.scatter(spike_idx, spike_values,
#                          color=colors['spikes'], s=80, zorder=3,
#                          edgecolor='white', linewidth=0.8)
#
#         # 优化刻度间隔
#         max_spike = np.max(spike_values)
#         ax_spike.set_yticks(np.arange(0, max_spike + 1, 1))
#         ax_spike.set_ylim(-0.1, max_spike + 0.2)
#     else:
#         ax_spike.set_ylim(-0.1, 0.5)
#         ax_spike.set_yticks([0, 0.5])
#
#     # 脉冲轴设置：x=0与y=0重叠
#     # 坐标轴设置修改
#     ax_spike.set_xlim(-0.2, T - 0.5)  # 原xlim(0, ...) → 左侧留-0.2余量
#     ax_spike.spines['bottom'].set_position(('data', 0))  # x轴仍在y=0
#     ax_spike.spines['left'].set_position(('data', -0.1))  # 原y轴在x=0 → 移至x=-0.1
#     ax_spike.set_xticks(range(0, T))
#     ax_spike.set_xticklabels([str(i) for i in range(0, T)])
#     ax_spike.set_xlabel('Time Step', fontsize=16, labelpad=10)
#     ax_spike.set_ylabel('Spike', fontsize=16, labelpad=10)
#     ax_spike.grid(axis='y', linestyle=':', color=colors['grid'], alpha=0.7)
#
#     plt.tight_layout()
#     plt.show()
# FIGURE D
#     import torch
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     import matplotlib.gridspec as gridspec
#     from scipy.stats import entropy, wasserstein_distance
#
#     # 设置字体支持
#     plt.rcParams["font.family"] = ["Times New Roman", "SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
#     plt.rcParams["axes.unicode_minus"] = False  # 确保负号正确显示
#
#     # -------------------------- 核心参数配置 --------------------------
#     np.random.seed(2025)
#     torch.manual_seed(2025)
#     T_total = 5  # 总时间步：输入T1-T5 → 膜电位T1-T5
#     n_samples = 1000  # 每个时间步的样本数量
#     input_show_idx = slice(0, 4)  # 绘图用输入：T1-T4（索引0-3）
#     mem_show_idx = slice(1, 5)  # 计算+绘图用膜电位：T2-T5（索引1-4）
#
#     # 输入分布参数 (均值, 标准差)
#     bn_params = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
#
#     # -------------------------- 生成输入数据（T1-T5） --------------------------
#     x_input = []
#     input_stats = []  # 存储输入的均值、方差（用于绘图标题）
#     for t in range(T_total):
#         mu, sigma = bn_params[t]
#         # 生成带微小噪声的正态分布输入
#         data = np.random.normal(mu, sigma, n_samples) + np.random.normal(0, 0.001, n_samples)
#         x_input.append(torch.tensor(data, dtype=torch.float32))
#         input_stats.append((np.mean(data), np.std(data) ** 2))
#
#     # -------------------------- 初始化神经元（对接已有的类） --------------------------
#     neuron_names = ["FR-LIF", "SR-LIF", "HR-LIF"]
#     # 直接使用已定义的神经元类（确保类含reset()、__call__()、u属性）
#     neurons = [ILIF(n=1), softLIF(), hardLIF()]
#
#     # -------------------------- 生成膜电位数据（T1-T5） --------------------------
#     all_mem_data = []  # 每个神经元的5个时间步膜电位分布（数组列表）
#     all_mem_stats = []  # 每个膜电位的均值、方差（用于绘图标题）
#     for neuron in neurons:
#         mem_t_list = [[] for _ in range(T_total)]  # 存储单个神经元T1-T5的膜电位
#         for sample_idx in range(n_samples):
#             neuron.reset()  # 每个样本独立初始化神经元状态
#             for t in range(T_total):
#                 x_t = x_input[t][sample_idx]  # 当前时间步的输入
#                 neuron(x_t)  # 计算膜电位（调用神经元类的__call__方法）
#                 mem_t_list[t].append(neuron.u.item())  # 记录当前膜电位值
#         # 转换为numpy数组并计算统计量
#         mem_array_list = [np.array(mem_list) for mem_list in mem_t_list]
#         mem_stats_list = [(np.mean(arr), np.var(arr)) for arr in mem_array_list]
#         all_mem_data.append(mem_array_list)
#         all_mem_stats.append(mem_stats_list)
#
#
#     # -------------------------- 分布距离计算（单独分箱策略） --------------------------
#     def get_smoothed_hist(data, bins, eps=1e-10):
#         """生成平滑概率直方图（避免0值导致log计算错误）"""
#         hist, _ = np.histogram(data, bins=bins, density=True)
#         hist = hist + eps  # 加微小值防止log(0)
#         return hist / np.sum(hist)  # 归一化为概率分布
#
#
#     # 1. 为每个神经元单独计算分箱（仅用自身膜电位数据，避免交叉干扰）
#     all_bins = []
#     for mem_list in all_mem_data:
#         # 展平当前神经元所有时间步的膜电位数据
#         mem_flat = np.concatenate(mem_list)
#         # 生成覆盖该神经元膜电位范围的50个分箱
#         bins = np.linspace(np.min(mem_flat), np.max(mem_flat), 50)
#         all_bins.append(bins)
#
#     # 2. 计算每个神经元的KL/JS/Wasserstein距离（时序前→后两两对比）
#     all_kl_avg = []  # 平均KL散度
#     all_js_avg = []  # 平均JS散度
#     all_wd_avg = []  # 平均Wasserstein距离
#     for i, mem_list in enumerate(all_mem_data):
#         mem_show = mem_list[mem_show_idx]  # 当前神经元的T2-T5膜电位（共4个时间步，索引0-3对应T2-T5）
#         current_bins = all_bins[i]
#         kl_list = []
#         js_list = []
#         wd_list = []
#
#         # 仅计算相邻时间步：T2-T3（0-1）、T3-T4（1-2）、T4-T5（2-3）
#         for t_prev in range(len(mem_show) - 1):  # t_prev取0、1、2（避免t_prev+1超出索引）
#             t_next = t_prev + 1  # 仅相邻后一个时间步
#             # 生成平滑概率分布
#             p_hist = get_smoothed_hist(mem_show[t_prev], current_bins)
#             q_hist = get_smoothed_hist(mem_show[t_next], current_bins)
#             # 计算各距离指标
#             kl = entropy(p_hist, q_hist)
#             kl_list.append(kl)
#             m_hist = (p_hist + q_hist) / 2
#             js = (entropy(p_hist, m_hist) + entropy(q_hist, m_hist)) / 2
#             js_list.append(js)
#             wd = wasserstein_distance(mem_show[t_prev], mem_show[t_next])
#             wd_list.append(wd)
#
#         # 计算相邻时间步的距离平均值
#         all_kl_avg.append(np.mean(kl_list) if kl_list else 0)
#         all_js_avg.append(np.mean(js_list) if js_list else 0)
#         all_wd_avg.append(np.mean(wd_list) if wd_list else 0)
#
#     # -------------------------- 计算统一坐标轴范围（修复切片遍历问题，确保曲线完整显示） --------------------------
#     # 将切片转换为可遍历的索引列表
#     input_show_indices = list(range(input_show_idx.start, input_show_idx.stop))  # input_show_idx → [0,1,2,3]
#     mem_show_indices = list(range(mem_show_idx.start, mem_show_idx.stop))  # mem_show_idx → [1,2,3,4]
#
#     # 1. 输入数据的全局范围（x轴统一，y轴按实际密度调整）
#     input_flat = np.concatenate([x.numpy() for x in x_input[input_show_idx]])
#     input_min, input_max = np.min(input_flat), np.max(input_flat)
#     input_xlim = (input_min - 0.1 * (input_max - input_min), input_max + 0.1 * (input_max - input_min))
#
#     # 修复：遍历索引列表而非切片对象，计算输入分布最大密度
#     input_max_density = 0
#     for t_idx in input_show_indices:  # 用转换后的列表遍历，避免TypeError
#         # 临时计算KDE密度最大值（不显示绘图）
#         kde = sns.kdeplot(x_input[t_idx].numpy(), fill=False, warn_singular=False)
#         if kde.get_lines():  # 确保有曲线数据
#             density_curve = kde.get_lines()[0].get_data()
#             current_max = np.max(density_curve[1])
#             if current_max > input_max_density:
#                 input_max_density = current_max
#         plt.clf()  # 清除临时绘图
#     input_ylim = (0, input_max_density * 1.1)  # 留10%余量
#
#     # 2. 每个神经元膜电位的范围（x轴统一，y轴按实际密度调整）
#     neuron_xlims = []
#     neuron_ylims = []
#     for mem_list in all_mem_data:
#         # x轴范围
#         mem_flat = np.concatenate([mem_list[idx] for idx in mem_show_indices])
#         mem_min, mem_max = np.min(mem_flat), np.max(mem_flat)
#         mem_xlim = (mem_min - 0.1 * (mem_max - mem_min), mem_max + 0.1 * (mem_max - mem_min))
#         neuron_xlims.append(mem_xlim)
#
#         # 修复：遍历索引列表，计算该神经元最大密度
#         mem_max_density = 0
#         for mem_idx in mem_show_indices:  # 用转换后的列表遍历
#             kde = sns.kdeplot(mem_list[mem_idx], fill=False, warn_singular=False)
#             if kde.get_lines():
#                 density_curve = kde.get_lines()[0].get_data()
#                 current_max = np.max(density_curve[1])
#                 if current_max > mem_max_density:
#                     mem_max_density = current_max
#             plt.clf()
#         mem_ylim = (0, mem_max_density * 1.1)
#         neuron_ylims.append(mem_ylim)
#
#     # -------------------------- 绘图：输入（T1-T4）+ 膜电位（T2-T5） --------------------------
#     fig = plt.figure(figsize=(21, 13))
#     gs = gridspec.GridSpec(4, 4, hspace=0.32, wspace=0.12)
#
#     # 颜色与字体配置（保持不变）
#     input_color = '#0A5E9C'
#     neuron_colors = ['#1A7F37', '#D46106', '#6A4C93']
#     font_title = {'family': 'Times New Roman', 'size': 19, 'weight': 'bold'}
#     font_label = {'family': 'Times New Roman', 'size': 18, 'weight': 'bold'}
#     font_tick = {'family': 'Times New Roman', 'size': 16}
#     font_kl = {'family': 'Times New Roman', 'size': 22, 'weight': 'bold'}
#
#     # 1. 第一行：输入分布（T1-T4）
#     for col in range(4):
#         ax = fig.add_subplot(gs[0, col])
#         t_idx = input_show_indices[col]  # 用索引列表取值
#         sns.kdeplot(x_input[t_idx].numpy(), fill=True, color=input_color, alpha=0.7, ax=ax)
#
#         # 三行标题
#         title_line1 = f'Input'
#         title_line2 = f'T={t_idx + 1}'
#         title_line3 = f'μ={input_stats[t_idx][0]:.3f}, σ²={input_stats[t_idx][1]:.3f}'
#         ax.text(0.02, 0.96, title_line1, transform=ax.transAxes, va='top', ha='left', fontdict=font_title, color='black')
#         ax.text(0.02, 0.86, title_line2, transform=ax.transAxes, va='top', ha='left', fontdict=font_title, color='black')
#         ax.text(0.02, 0.76, title_line3, transform=ax.transAxes, va='top', ha='left', fontdict=font_title, color='black')
#
#         # 仅第一列显示y轴标签
#         if col == 0:
#             ax.set_ylabel('Density', fontdict=font_label)
#         else:
#             ax.set_ylabel('')
#
#         ax.set_xlabel('Input Current', fontdict=font_label)
#         ax.grid(alpha=0.3, linestyle='--', linewidth=1.0)
#         ax.tick_params(axis='both', labelsize=font_tick['size'], pad=5)
#         ax.set_xlim(input_xlim)
#         ax.set_ylim(input_ylim)  # 使用优化后的y轴范围
#
#     # 2. 第2-4行：膜电位分布（T2-T5）
#     for row, (name, mem_list, mem_stats, xlim, ylim, kl_avg) in enumerate(
#             zip(neuron_names, all_mem_data, all_mem_stats, neuron_xlims, neuron_ylims, all_kl_avg)):
#         for col in range(4):
#             ax = fig.add_subplot(gs[row + 1, col])
#             mem_idx = mem_show_indices[col]  # 用索引列表取值
#             sns.kdeplot(mem_list[mem_idx], fill=True, color=neuron_colors[row], alpha=0.7, ax=ax)
#
#             # 三行标题
#             title_line1 = f'{name}'
#             title_line2 = f'T={mem_idx + 1}'
#             title_line3 = f'μ={mem_stats[mem_idx][0]:.3f}, σ²={mem_stats[mem_idx][1]:.3f}'
#             ax.text(0.02, 0.96, title_line1, transform=ax.transAxes, va='top', ha='left', fontdict=font_title,
#                     color='black')
#             ax.text(0.02, 0.86, title_line2, transform=ax.transAxes, va='top', ha='left', fontdict=font_title,
#                     color='black')
#             ax.text(0.02, 0.76, title_line3, transform=ax.transAxes, va='top', ha='left', fontdict=font_title,
#                     color='black')
#
#             # 仅第一列显示y轴标签
#             if col == 0:
#                 ax.set_ylabel('Density', fontdict=font_label)
#             else:
#                 ax.set_ylabel('')
#
#             # KL散度显示
#             if col == 0:
#                 ax.text(0.02, 0.08,
#                         f'Avg KL Divergence: {kl_avg:.4f}',
#                         transform=ax.transAxes,
#                         verticalalignment='bottom',
#                         horizontalalignment='left',
#                         fontdict=font_kl,
#                         color='crimson')
#
#             ax.set_xlabel('Membrane Potential', fontdict=font_label)
#             ax.grid(alpha=0.3, linestyle='--', linewidth=1.0)
#             ax.tick_params(axis='both', labelsize=font_tick['size'], pad=5)
#             ax.set_xlim(xlim)
#             ax.set_ylim(ylim)  # 使用优化后的y轴范围
#
#     # 调整边距
#     plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.07)
#     # 保存为矢量图
#     plt.savefig('membrane_potential_distribution_analysis_final_v8.pdf',
#                 dpi=300,
#                 bbox_inches='tight',
#                 pad_inches=0.15)
#     plt.show()
#
#     # -------------------------- 打印分布距离指标结果 --------------------------
#     print("=== 膜电位后4个时间步（T2-T5）分布距离指标（单独分箱） ===")
#     for name, kl, js, wd in zip(neuron_names, all_kl_avg, all_js_avg, all_wd_avg):
#         print(f"\n{name}:")
#         print(f"  平均KL散度（前→后时序）: {kl:.4f}")
#         print(f"  平均JS散度: {js:.4f}")
#         print(f"  平均Wasserstein距离: {wd:.4f}")

# # FIGURE E
#     import torch
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     import matplotlib.gridspec as gridspec
#     from scipy.stats import entropy, wasserstein_distance, gaussian_kde
#     import os
#
#     # 设置字体支持
#     plt.rcParams["font.family"] = ["Times New Roman", "SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
#     plt.rcParams["axes.unicode_minus"] = False  # 确保负号正确显示
#     sns.set_style("whitegrid")  # 简洁网格风格
#
#     # -------------------------- 核心参数配置 --------------------------
#     np.random.seed(2025)
#     torch.manual_seed(2025)
#     T_total = 4  # 总时间步：输入T1-T4 → 膜电位T1-T4
#     n_samples = 1000  # 每个时间步的样本数量
#     input_show_idx = slice(0, 3)  # 绘图用输入：T1-T3（索引0-2）
#     mem_show_idx = slice(1, 4)  # 计算+绘图用膜电位：T2-T4（索引1-3）
#
#     # 输入分布参数 (均值, 标准差)
#     bn_params = [(0, 1), (0, 1), (0, 1), (0, 1)]
#
#     # -------------------------- 生成输入数据（T1-T4） --------------------------
#     x_input = []
#     input_stats = []  # 存储输入的均值、方差（用于绘图标题）
#     for t in range(T_total):
#         mu, sigma = bn_params[t]
#         data = np.random.normal(mu, sigma, n_samples) + np.random.normal(0, 0.2, n_samples)
#         x_input.append(torch.tensor(data, dtype=torch.float32))
#         input_stats.append((np.mean(data), np.std(data) ** 2))
#
#     # 验证输入范围
#     input_min_all = min([np.min(x.numpy()) for x in x_input])
#     input_max_all = max([np.max(x.numpy()) for x in x_input])
#     print(f"=== 输入数据范围验证 ===")
#     print(f"输入全局最小值：{input_min_all:.4f}，全局最大值：{input_max_all:.4f}")
#     print(f"结论：输入(0,1)分布下，99.7%样本落在[-3,3]，几乎无大于4的值\n")
#
#     # -------------------------- 初始化神经元（不同n的FR-LIIF） --------------------------
#     n_values = [1, 2, 3, 4]  # n值：1,2,4,8
#     neuron_names = [f"FR-LIMF (n={n})" for n in n_values]  # 神经元名称（明确n值）
#     neurons = [ILIF(n=n) for n in n_values]  # 初始化4个不同n的FR-LIIF神经元
#
#     # -------------------------- 生成膜电位数据（T1-T4） --------------------------
#     all_mem_data = []  # 每个神经元的4个时间步膜电位分布
#     all_mem_stats = []  # 每个膜电位的均值、方差
#     for neuron in neurons:
#         mem_t_list = [[] for _ in range(T_total)]  # 存储单个神经元T1-T4的膜电位
#         for sample_idx in range(n_samples):
#             neuron.reset()  # 每个样本独立初始化
#             for t in range(T_total):
#                 x_t = x_input[t][sample_idx]  # 当前时间步输入
#                 neuron(x_t)  # 计算膜电位
#                 mem_t_list[t].append(neuron.u.item())  # 记录膜电位
#         # 转换为numpy数组并计算统计量
#         mem_array_list = [np.array(mem_list) for mem_list in mem_t_list]
#         mem_stats_list = [(np.mean(arr), np.var(arr)) for arr in mem_array_list]
#         all_mem_data.append(mem_array_list)
#         all_mem_stats.append(mem_stats_list)
#
#
#     # -------------------------- 分布距离计算（单独分箱策略） --------------------------
#     def get_smoothed_hist(data, bins, eps=1e-10):
#         """生成平滑概率直方图（避免0值导致log计算错误）"""
#         hist, _ = np.histogram(data, bins=bins, density=True)
#         hist = hist + eps  # 加微小值防止log(0)
#         return hist / np.sum(hist)  # 归一化为概率分布
#
#
#     # 1. 为每个神经元单独计算分箱
#     all_bins = []
#     for mem_list in all_mem_data:
#         mem_flat = np.concatenate(mem_list)
#         bins = np.linspace(np.min(mem_flat), np.max(mem_flat), 50)
#         all_bins.append(bins)
#
#     # 2. 计算相邻时间步的距离指标
#     all_kl_avg = []
#     all_js_avg = []
#     all_wd_avg = []
#     for i, mem_list in enumerate(all_mem_data):
#         mem_show = mem_list[mem_show_idx]  # T2-T4膜电位
#         current_bins = all_bins[i]
#         kl_list = []
#         js_list = []
#         wd_list = []
#
#         # 仅计算相邻时间步（T2-T3、T3-T4）
#         for t_prev in range(len(mem_show) - 1):
#             t_next = t_prev + 1
#             p_hist = get_smoothed_hist(mem_show[t_prev], current_bins)
#             q_hist = get_smoothed_hist(mem_show[t_next], current_bins)
#
#             # 计算各距离指标
#             kl = entropy(p_hist, q_hist)
#             kl_list.append(kl)
#             m_hist = (p_hist + q_hist) / 2
#             js = (entropy(p_hist, m_hist) + entropy(q_hist, m_hist)) / 2
#             js_list.append(js)
#             wd = wasserstein_distance(mem_show[t_prev], mem_show[t_next])
#             wd_list.append(wd)
#
#         # 计算平均值
#         all_kl_avg.append(np.mean(kl_list) if kl_list else 0)
#         all_js_avg.append(np.mean(js_list) if js_list else 0)
#         all_wd_avg.append(np.mean(wd_list) if wd_list else 0)
#
#     # -------------------------- 计算统一坐标轴范围 --------------------------
#     mem_show_indices = list(range(mem_show_idx.start, mem_show_idx.stop))  # slice→索引列表
#
#     # 1. 输入数据全局范围（T1-T3）
#     input_flat = np.concatenate([x.numpy() for x in x_input[input_show_idx]])
#     input_min, input_max = np.min(input_flat), np.max(input_flat)
#     input_xlim = (input_min - 0.1 * (input_max - input_min), input_max + 0.1 * (input_max - input_min))
#
#     # 2. 每个神经元膜电位全局范围（T2-T4）
#     neuron_xlims = []
#     for mem_list in all_mem_data:
#         mem_flat = np.concatenate([mem_list[idx] for idx in mem_show_indices])
#         mem_min, mem_max = np.min(mem_flat), np.max(mem_flat)
#         mem_xlim = (mem_min - 0.1 * (mem_max - mem_min), mem_max + 0.1 * (mem_max - mem_min))
#         neuron_xlims.append(mem_xlim)
#
#     # -------------------------- 绘图核心修改 --------------------------
#     fig = plt.figure(figsize=(18, 20))  # 纵向尺寸增大2，适配更大纵向间距
#     gs = gridspec.GridSpec(
#         5, 3,
#         hspace=0.35,  # 核心：纵向间距从0.25→0.35，解决横坐标遮挡
#         wspace=0.15,  # 保持压缩后的横向间距
#         figure=fig,
#         left=0.06,
#         right=0.94,
#         top=0.95,
#         bottom=0.05
#     )
#
#     # 专业配色方案（保持不变）
#     input_color = '#2E86AB'  # 输入：深蓝色
#     neuron_colors = [
#         '#A23B72',  # n=1：深紫红
#         '#F18F01',  # n=2：暖橙色
#         '#C73E1D',  # n=4：深红色
#         '#3F88C5'  # n=8：天蓝色
#     ]
#     peak_line_colors = [
#         '#7B2953',  # n=1：深紫红深色版
#         '#C07000',  # n=2：暖橙色深色版
#         '#9A2E13',  # n=4：深红色深色版
#         '#2A6BA3'  # n=8：天蓝色深色版
#     ]
#     peak_text_color = 'black'
#
#     # 字体配置（保持大幅增大规格）
#     title_font = {'family': 'Times New Roman', 'size': 24, 'weight': 'bold', 'color': 'black'}
#     kl_font = {'family': 'Times New Roman', 'size': 22, 'weight': 'bold', 'color': '#E74C3C'}
#     font_label = {'family': 'Times New Roman', 'size': 22, 'weight': 'bold'}
#     font_tick = {'family': 'Times New Roman', 'size': 20}
#     peak_font = {'family': 'Times New Roman', 'size': 20, 'weight': 'bold'}
#
#     # 1. 第一行：输入分布（T1-T3）
#     for col in range(3):
#         ax = fig.add_subplot(gs[0, col])
#         t_idx = input_show_idx.start + col  # 0-2 → T1-T3
#         sns.kdeplot(
#             x_input[t_idx].numpy(),
#             fill=True, color=input_color, alpha=0.8, ax=ax,
#             linewidth=2, edgecolor='black'
#         )
#
#         # 核心1：去掉标题背景框，纯文字显示
#         title_text = f'Input T={t_idx + 1}\nμ={input_stats[t_idx][0]:.3f}\nσ²={input_stats[t_idx][1]:.3f}'
#         ax.text(
#             0.03, 0.97,
#             title_text,
#             transform=ax.transAxes,
#             fontdict=title_font,
#             verticalalignment='top',
#             horizontalalignment='left',
#             zorder=5
#         )
#
#         # 仅第一列显示y轴标签
#         ax.set_xlabel('Input Current', fontdict=font_label)
#         if col == 0:
#             ax.set_ylabel('Density', fontdict=font_label)
#         else:
#             ax.set_ylabel('')
#
#         ax.grid(alpha=0.4, linestyle='--', linewidth=0.8)
#         ax.tick_params(axis='both', labelsize=font_tick['size'], pad=6)
#         ax.set_xlim(input_xlim)
#         # 缩小曲线占比（y轴上限=峰值×1.1）
#         kde = gaussian_kde(x_input[t_idx].numpy())
#         x_range = np.linspace(input_xlim[0], input_xlim[1], 1000)
#         peak_y = np.max(kde(x_range))
#         ax.set_ylim(0, peak_y * 1.1)
#         ax.xaxis.labelpad = 12  # 增大x轴标签间距，进一步避免遮挡
#         ax.yaxis.labelpad = 12
#
#     # 2. 第2-5行：膜电位分布
#     for row, (name, mem_list, mem_stats, xlim, color, peak_line_color, kl_avg) in enumerate(
#             zip(neuron_names, all_mem_data, all_mem_stats, neuron_xlims, neuron_colors, peak_line_colors, all_kl_avg)
#     ):
#         for col in range(3):
#             ax = fig.add_subplot(gs[row + 1, col])
#             mem_idx = mem_show_idx.start + col  # 1-3 → T2-T4
#             mem_data = mem_list[mem_idx]
#
#             # 绘制KDE分布
#             sns.kdeplot(
#                 mem_data,
#                 fill=True, color=color, alpha=0.8, ax=ax,
#                 linewidth=2, edgecolor='black'
#             )
#
#             # 计算峰值
#             kde = gaussian_kde(mem_data)
#             x_range = np.linspace(xlim[0], xlim[1], 1000)
#             kde_values = kde(x_range)
#             peak_x = x_range[np.argmax(kde_values)]
#             peak_y = np.max(kde_values)
#
#             # 核心2：Peak标注移到右侧（不遮挡曲线，不占顶部空间）
#             ax.axvline(x=peak_x, color=peak_line_color, linestyle='--', linewidth=2.5, alpha=0.9)
#             ax.text(
#                 0.97, 0.5,  # 右侧垂直居中位置
#                 f'Peak: {peak_x:.3f}',
#                 ha='right', va='center',
#                 fontdict=peak_font,
#                 color=peak_text_color,
#                 rotation=90,  # 垂直文字，节省横向空间
#                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
#                 zorder=4,
#                 transform=ax.transAxes
#             )
#
#             # 核心1：去掉标题背景框，纯文字显示
#             title_text = f'{name}\nT={mem_idx + 1}\nμ={mem_stats[mem_idx][0]:.3f}\nσ²={mem_stats[mem_idx][1]:.3f}'
#             ax.text(
#                 0.03, 0.97,
#                 title_text,
#                 transform=ax.transAxes,
#                 fontdict=title_font,
#                 verticalalignment='top',
#                 horizontalalignment='left',
#                 zorder=5
#             )
#
#             # 第一列子图：左下角红色KL标注
#             if col == 0:
#                 kl_text = f'Avg KL Divergence:\n{kl_avg:.4f}'
#                 ax.text(
#                     0.03, 0.03,
#                     kl_text,
#                     transform=ax.transAxes,
#                     fontdict=kl_font,
#                     verticalalignment='bottom',
#                     horizontalalignment='left',
#                     bbox=dict(boxstyle='round,pad=0.4', facecolor='#FADBD8', alpha=0.9),
#                     zorder=5
#                 )
#
#             # 仅第一列显示y轴标签
#             ax.set_xlabel('Membrane Potential', fontdict=font_label)
#             if col == 0:
#                 ax.set_ylabel('Density', fontdict=font_label)
#             else:
#                 ax.set_ylabel('')
#
#             ax.grid(alpha=0.4, linestyle='--', linewidth=0.8)
#             ax.tick_params(axis='both', labelsize=font_tick['size'], pad=6)
#             ax.set_xlim(xlim)
#             # 缩小曲线占比（y轴上限=峰值×1.1）
#             ax.set_ylim(0, peak_y * 1.1)
#             ax.xaxis.labelpad = 12
#             ax.yaxis.labelpad = 12
#
#     # -------------------------- 保存为矢量图 --------------------------
#     save_filename = 'figure_e_optimized_final_v2.pdf'
#     save_success = False
#
#     # 尝试保存为PDF（优先选择）
#     try:
#         plt.savefig(
#             save_filename,
#             dpi=300,
#             bbox_inches='tight',
#             pad_inches=0.15,
#             format='pdf',
#             facecolor='white',
#             edgecolor='none'
#         )
#         save_success = True
#     except Exception as e:
#         print(f"PDF保存失败：{str(e)[:100]}")
#         # 备用方案：保存为SVG
#         save_filename = 'figure_e_optimized_final_v2.svg'
#         try:
#             plt.savefig(
#                 save_filename,
#                 dpi=300,
#                 bbox_inches='tight',
#                 pad_inches=0.15,
#                 format='svg'
#             )
#             save_success = True
#         except Exception as e2:
#             print(f"SVG保存也失败：{str(e2)[:100]}")
#
#     plt.close(fig)
#
#     # -------------------------- 结果输出 --------------------------
#     print("=== 膜电位3个时间步（T2-T4）分布距离指标（单独分箱） ===")
#     print("\n【关键说明】")
#     print(f"1. 输入(0,1)分布的样本几乎无大于4的值（最大仅{input_max_all:.2f}），导致n=4和n=8的FR-LIIF神经元：")
#     print("   - 输出上限未被触发（神经元允许的最大输出>4，但输入未达到触发阈值）")
#     print("   - 膜电位分布缺乏差异化，表现为距离指标接近")
#     print("\n2. 若需让n=4/8显现差异，可尝试：")
#     print("   - 调整输入分布：将标准差改为3（bn_params=[(0,3),...]），使输入最大值超过4")



# for ar-lif
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.gridspec as gridspec
    from scipy.stats import entropy, wasserstein_distance

    # 设置字体支持
    plt.rcParams["font.family"] = ["Times New Roman", "SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams["axes.unicode_minus"] = False  # 确保负号正确显示

    # -------------------------- 核心参数配置 --------------------------
    np.random.seed(2025)
    torch.manual_seed(2025)
    T_total = 5  # 总时间步：输入T1-T5 → 膜电位T1-T5
    n_samples = 2000  # 每个时间步的样本数量
    input_show_idx = slice(0, 4)  # 绘图用输入：T1-T4（索引0-3）
    mem_show_idx = slice(1, 5)  # 计算+绘图用膜电位：T2-T5（索引1-4）

    # 输入分布参数 (均值, 标准差)
    bn_params = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]

    # -------------------------- 生成输入数据（T1-T5） --------------------------
    x_input = []
    input_stats = []  # 存储输入的均值、方差（用于绘图标题）
    for t in range(T_total):
        mu, sigma = bn_params[t]
        # 生成带微小噪声的正态分布输入
        data = np.random.normal(mu, sigma, n_samples) + np.random.normal(0, 0.1, n_samples)
        x_input.append(torch.tensor(data, dtype=torch.float32))
        input_stats.append((np.mean(data), np.std(data) ** 2))

    # -------------------------- 初始化神经元（对接已有的类） --------------------------
    neuron_names = ["AR-LIF", "SR-LIF", "HR-LIF"]
    # 直接使用已定义的神经元类（确保类含 reset()、__call__()、u 属性）
    neurons = [XLIF(), softLIF(), hardLIF()]

    # -------------------------- 生成膜电位数据（T1-T5） --------------------------
    all_mem_data = []  # 每个神经元的5个时间步膜电位分布（数组列表）
    all_mem_stats = []  # 每个膜电位的均值、方差（用于绘图标题）
    for neuron in neurons:
        mem_t_list = [[] for _ in range(T_total)]  # 存储单个神经元T1-T5的膜电位
        for sample_idx in range(n_samples):
            neuron.reset()  # 每个样本独立初始化神经元状态
            for t in range(T_total):
                x_t = x_input[t][sample_idx]  # 当前时间步的输入
                neuron(x_t)  # 计算膜电位（调用神经元类的 __call__ 方法）
                mem_t_list[t].append(neuron.u.item())  # 记录当前膜电位值
        # 转换为 numpy 数组并计算统计量
        mem_array_list = [np.array(mem_list) for mem_list in mem_t_list]
        mem_stats_list = [(np.mean(arr), np.var(arr)) for arr in mem_array_list]
        all_mem_data.append(mem_array_list)
        all_mem_stats.append(mem_stats_list)


    # -------------------------- 分布距离计算（单独分箱策略） --------------------------
    def get_smoothed_hist(data, bins, eps=1e-10):
        """生成平滑概率直方图（避免0值导致log计算错误）"""
        hist, _ = np.histogram(data, bins=bins, density=True)
        hist = hist + eps  # 加微小值防止log(0)
        return hist / np.sum(hist)  # 归一化为概率分布


    # 1. 为每个神经元单独计算分箱（仅用自身膜电位数据，避免交叉干扰）
    all_bins = []
    for mem_list in all_mem_data:
        # 展平当前神经元所有时间步的膜电位数据
        mem_flat = np.concatenate(mem_list)
        # 生成覆盖该神经元膜电位范围的50个分箱
        bins = np.linspace(np.min(mem_flat), np.max(mem_flat), 50)
        all_bins.append(bins)

    # 2. 计算每个神经元的 KL / JS / Wasserstein 距离（时序前→后两两对比）
    all_kl_avg = []  # 平均 KL 散度
    all_js_avg = []  # 平均 JS 散度
    all_wd_avg = []  # 平均 Wasserstein 距离
    for i, mem_list in enumerate(all_mem_data):
        mem_show = mem_list[mem_show_idx]  # 当前神经元的 T2-T5 膜电位（共4个时间步，索引0-3对应T2-T5）
        current_bins = all_bins[i]
        kl_list = []
        js_list = []
        wd_list = []

        # 仅计算相邻时间步：T2-T3（0-1）、T3-T4（1-2）、T4-T5（2-3）
        for t_prev in range(len(mem_show) - 1):  # t_prev 取 0、1、2
            t_next = t_prev + 1  # 仅相邻后一个时间步
            # 生成平滑概率分布
            p_hist = get_smoothed_hist(mem_show[t_prev], current_bins)
            q_hist = get_smoothed_hist(mem_show[t_next], current_bins)

            # KL
            kl = entropy(p_hist, q_hist)
            kl_list.append(kl)

            # JS
            m_hist = (p_hist + q_hist) / 2
            js = (entropy(p_hist, m_hist) + entropy(q_hist, m_hist)) / 2
            js_list.append(js)

            # Wasserstein
            wd = wasserstein_distance(mem_show[t_prev], mem_show[t_next])
            wd_list.append(wd)

        # 计算相邻时间步的距离平均值
        all_kl_avg.append(np.mean(kl_list) if kl_list else 0.0)
        all_js_avg.append(np.mean(js_list) if js_list else 0.0)
        all_wd_avg.append(np.mean(wd_list) if wd_list else 0.0)

    # -------------------------- 统一 x 轴范围（但不强行统一 y 轴） --------------------------
    # 将切片转换为可遍历的索引列表
    input_show_indices = list(range(input_show_idx.start, input_show_idx.stop))  # [0,1,2,3]
    mem_show_indices = list(range(mem_show_idx.start, mem_show_idx.stop))  # [1,2,3,4]

    # 1. 输入数据的全局 x 范围
    input_flat = np.concatenate([x.numpy() for x in x_input[input_show_idx]])
    input_min, input_max = np.min(input_flat), np.max(input_flat)
    input_xlim = (input_min - 0.1 * (input_max - input_min),
                  input_max + 0.1 * (input_max - input_min))

    # 2. 每个神经元膜电位的 x 范围
    neuron_xlims = []
    for mem_list in all_mem_data:
        mem_flat = np.concatenate([mem_list[idx] for idx in mem_show_indices])
        mem_min, mem_max = np.min(mem_flat), np.max(mem_flat)
        mem_xlim = (mem_min - 0.1 * (mem_max - mem_min),
                    mem_max + 0.1 * (mem_max - mem_min))
        neuron_xlims.append(mem_xlim)

    # -------------------------- 绘图：输入（T1-T4）+ 膜电位（T2-T5） --------------------------
    # 关键修改1：整张图改成接近正方形
    fig = plt.figure(figsize=(16, 16))
    gs = gridspec.GridSpec(4, 4, hspace=0.32, wspace=0.12)

    # 配色
    input_color = '#2E86AB'  # 输入颜色
    neuron_colors = ['#1B9C85', '#E67E22', '#8E44AD']  # AR / SR / HR 各一色

    font_title = {'family': 'Times New Roman', 'size': 19, 'weight': 'bold'}
    font_label = {'family': 'Times New Roman', 'size': 18, 'weight': 'bold'}
    font_tick = {'family': 'Times New Roman', 'size': 16}
    font_metric = {'family': 'Times New Roman', 'size': 20, 'weight': 'bold'}

    # 1. 第一行：输入分布（T1-T4）
    for col in range(4):
        ax = fig.add_subplot(gs[0, col])
        ax.set_box_aspect(1)  # 关键修改2：每个小图强制正方形
        t_idx = input_show_indices[col]
        sns.kdeplot(x_input[t_idx].numpy(), fill=True, color=input_color, alpha=0.7, ax=ax)

        # 三行标题
        title_line1 = f'Input'
        title_line2 = f'T={t_idx + 1}'
        title_line3 = f'μ={input_stats[t_idx][0]:.3f}, σ²={input_stats[t_idx][1]:.3f}'
        ax.text(0.02, 0.96, title_line1, transform=ax.transAxes, va='top', ha='left',
                fontdict=font_title, color='black')
        ax.text(0.02, 0.86, title_line2, transform=ax.transAxes, va='top', ha='left',
                fontdict=font_title, color='black')
        ax.text(0.02, 0.76, title_line3, transform=ax.transAxes, va='top', ha='left',
                fontdict=font_title, color='black')

        if col == 0:
            ax.set_ylabel('Density', fontdict=font_label)
        else:
            ax.set_ylabel('')

        ax.set_xlabel('Input Current', fontdict=font_label)
        ax.grid(alpha=0.3, linestyle='--', linewidth=1.0)
        ax.tick_params(axis='both', labelsize=font_tick['size'], pad=5)
        ax.set_xlim(input_xlim)

    # 2. 第2-4行：膜电位分布（T2-T5）
    for row, (name, mem_list, mem_stats, xlim, js_avg, wd_avg) in enumerate(
            zip(neuron_names, all_mem_data, all_mem_stats, neuron_xlims, all_js_avg, all_wd_avg)):
        for col in range(4):
            ax = fig.add_subplot(gs[row + 1, col])
            ax.set_box_aspect(1)  # 关键修改2：每个小图强制正方形
            mem_idx = mem_show_indices[col]
            sns.kdeplot(mem_list[mem_idx], fill=True, color=neuron_colors[row], alpha=0.7, ax=ax)

            # 三行标题
            title_line1 = f'{name}'
            title_line2 = f'T={mem_idx + 1}'
            title_line3 = f'μ={mem_stats[mem_idx][0]:.3f}, σ²={mem_stats[mem_idx][1]:.3f}'
            ax.text(0.02, 0.96, title_line1, transform=ax.transAxes, va='top', ha='left',
                    fontdict=font_title, color='black')
            ax.text(0.02, 0.86, title_line2, transform=ax.transAxes, va='top', ha='left',
                    fontdict=font_title, color='black')
            ax.text(0.02, 0.76, title_line3, transform=ax.transAxes, va='top', ha='left',
                    fontdict=font_title, color='black')

            # 仅第一列显示 y 轴标签
            if col == 0:
                ax.set_ylabel('Density', fontdict=font_label)
            else:
                ax.set_ylabel('')

            # 指标显示：JS + Wasserstein，两行展示（每个神经元左侧一格）
            if col == 0:
                ax.text(0.02, 0.16,
                        f'Avg JS: {js_avg:.4f}',
                        transform=ax.transAxes,
                        va='bottom', ha='left',
                        fontdict=font_metric,
                        color='crimson')
                ax.text(0.02, 0.06,
                        f'Avg W:  {wd_avg:.4f}',
                        transform=ax.transAxes,
                        va='bottom', ha='left',
                        fontdict=font_metric,
                        color='crimson')

            ax.set_xlabel('Membrane Potential', fontdict=font_label)
            ax.grid(alpha=0.3, linestyle='--', linewidth=1.0)
            ax.tick_params(axis='both', labelsize=font_tick['size'], pad=5)
            ax.set_xlim(xlim)

    # 调整边距
    plt.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.06)
    # 保存为矢量图
    plt.savefig('membrane_potential_distribution_analysis_square_subplots.pdf',
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.15)
    plt.show()

    # -------------------------- 打印分布距离指标结果 --------------------------
    print("=== 膜电位后4个时间步（T2-T5）分布距离指标（单独分箱） ===")
    for name, kl, js, wd in zip(neuron_names, all_kl_avg, all_js_avg, all_wd_avg):
        print(f"\n{name}:")
        print(f"  平均JS散度（前→后时序）: {js:.4f}")
        print(f"  平均Wasserstein距离:     {wd:.4f}")
        print(f"  （参考）平均KL散度:     {kl:.4f}")
