import math

import torch
import torch.nn as nn

from spikingjelly.clock_driven.surrogate import SurrogateFunctionBase, heaviside
from torch.cuda.amp import autocast


class rectangle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, vth):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.vth = vth
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            x = ctx.saved_tensors[0]
            mask1 = (x.abs() > ctx.vth / 2)
            mask_ = mask1.logical_not()
            grad_x = grad_output * x.masked_fill(mask_, 1. / ctx.vth).masked_fill(mask1, 0.)
        return grad_x, None


class Rectangle(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return rectangle.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return torch.min(torch.max(1. / alpha * x, 0.5), -0.5)

class rectangle2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, vth):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.vth = vth
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            x = ctx.saved_tensors[0]
            mask1 = (x.abs() > ctx.vth / 2)
            mask_ = mask1.logical_not()
            grad_x = grad_output * x.masked_fill(mask_, 1. / ctx.vth).masked_fill(mask1, 0.)
        return grad_x, None


class Rectangle2(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return rectangle2.apply(x, 2)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return torch.min(torch.max(1. / alpha * x, 0.5), -0.5)

class rectangle3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, vth):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.vth = vth
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            x = ctx.saved_tensors[0]
            mask1 = (x.abs() > ctx.vth / 2)
            mask_ = mask1.logical_not()
            grad_x = grad_output * x.masked_fill(mask_, 1. / ctx.vth).masked_fill(mask1, 0.)
        return grad_x, None


class Rectangle3(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return rectangle3.apply(x, 3)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return torch.min(torch.max(1. / alpha * x, 0.5), -0.5)

class rectangle4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, vth):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.vth = vth
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            x = ctx.saved_tensors[0]
            mask1 = (x.abs() > ctx.vth / 2)
            mask_ = mask1.logical_not()
            grad_x = grad_output * x.masked_fill(mask_, 1. / ctx.vth).masked_fill(mask1, 0.)
        return grad_x, None


class Rectangle4(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return rectangle4.apply(x, 4)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return torch.min(torch.max(1. / alpha * x, 0.5), -0.5)

class ZIF(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return zif.apply(x, alpha)

class zif(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None

# class Clip(nn.Module):
#     def __init__(self):
#         super(Clip, self).__init__()
#     class quant4(torch.autograd.Function):
#         @staticmethod
#         def forward(ctx, input, param):
#             # 前向计算：逐元素除法 → 取整 → 钳位（同尺寸操作）
#             y = input / param  # 逐元素: input[i,j,k,l]/param[i,j,k,l]
#             floor_y = torch.floor(y)
#             output = torch.clamp(floor_y, min=0, max=4)
#             # 保存反向所需的张量（均为同尺寸）
#             ctx.save_for_backward(input, param)
#             ctx.floor_y = floor_y
#             return output
#         @staticmethod
#         def backward(ctx, grad_output):
#             input, param = ctx.saved_tensors
#             # TEST benshen xiaoguo
#             # lower_bounds = 1 * param - 0.5
#             # upper_bounds = 16 * param + 0.5
#             # valid_mask = ((input >= lower_bounds) & (input <= upper_bounds))
#             # grad_output与input同尺寸（反向传播的梯度尺寸匹配）
#             assert grad_output.shape == input.shape, "grad_output must match input shape"
#             # 生成n=1~16，形状为 [16, 1, 1, 1, 1]，确保与input/param的所有维度可广播
#             # 维度数量 = input维度 + 1（n的维度），且n的非1维度在最前面
#             n = torch.arange(1, 1 + 4, device=input.device)  # 形状 [16]
#             # 扩展n的维度，使其与input的每个维度都有一个对应1的维度
#             n = n.view(-1, *([1] * input.dim()))  # 对于4D input，形状变为 [16, 1, 1, 1, 1]
#             # 无需扩展input和param，直接利用广播机制 # input形状 [B, C, H, W] 会自动广播为 [16, B, C, H, W] # param同理
#             lower_bounds = n * param - 0.5
#             upper_bounds = n * param + 0.5
#             # 计算掩码：input在任何n对应的区间内即有效 # 结果形状 [16, B, C, H, W]，沿n的维度合并
#             valid_mask = ((input >= lower_bounds) & (input <= upper_bounds)).any(dim=0)  # 合并后形状 [B, C, H, W]
#             # 验证掩码维度是否与input一致（关键检查）
#             assert valid_mask.shape == input.shape, f"掩码维度不匹配: {valid_mask.shape} vs {input.shape}"
#             # 1. grad_input：逐元素矩形替代梯度（与原逻辑一致，适配同尺寸param）
#             # 逐元素计算有效区域：input ∈ [param-0.5, param+0.5]
#             # valid_mask = (input >= (1 * param - 0.5)) & (input <= (16 * param + 0.5))  # 同尺寸bool张量
#             # 2025/10/13 debug：缺少 1 / vth！
#             # 确保param不会小于某个最小值
#             # safe_param = torch.clamp(torch.abs(param), min=1e-4)
#             grad_input = grad_output.clone() / param
#             grad_input = grad_input * valid_mask.float() # 有效区域保留梯度，无效区域置0
#             # 2. grad_param：逐元素梯度计算（适配同尺寸param）
#             # 仅在有效区域计算梯度，公式：grad_output * (-input) / (param²)
#             grad_param = torch.where(
#                 valid_mask,  # 逐元素判断有效区域
#                 grad_output * (-input) / (param ** 2),  # 逐元素梯度公式
#                 torch.tensor(0.0, device=param.device)  # 无效区域梯度为0
#             )
#             # grad_param与param同尺寸，无需聚合（逐元素对应更新）
#             return grad_input, grad_param
#     #
#     def forward(self, x, param):
#         return self.quant4.apply(x, param)

class Clip(nn.Module):
    def __init__(self, mode=1, n=1):
        """
        初始化可配置量化区间的Clip模块，支持三种梯度计算模式
        核心改进：将原固定的16级量化改为外部可配置参数n，适配不同量化需求

        Args:
            mode: 梯度计算模式（默认1）
                1: 原始模式（子区间并集）- 计算n个离散子区间 [k×param±0.5]（k=1~n）的并集，精度最高
                2: 连续区间模式（快速）- 单个连续区间 [param-0.5, n×param+0.5]，覆盖所有子区间范围，计算高效
                3: 固定区间模式（最快）- 用1替换param，固定区间 [0.5, n+0.5]，无需参数依赖
            n: 量化等级数（默认1）- 控制区间数量/范围，对应原代码中的16，需为正整数
        """
        super(Clip, self).__init__()
        # 校验输入参数合法性
        assert mode in [1, 2, 3], f"模式必须是1、2、3中的一种，当前输入：{mode}"
        assert isinstance(n, int) and n > 0, f"n必须是正整数，当前输入：{n}"
        self.mode = mode  # 保存梯度计算模式
        self.n = n  # 保存量化等级数（控制区间范围）

    class quant4(torch.autograd.Function):
        """
        自定义自动求导函数：实现量化的前向计算和自定义反向梯度传播
        前向：输入/param → 取整 → 钳位到[0, n]
        反向：根据不同模式计算有效梯度区域，实现逐元素梯度传播
        """

        @staticmethod
        def forward(ctx, input, param, mode, n):
            """
            前向传播：逐元素量化操作
            Args:
                ctx: 上下文对象，用于保存反向传播所需数据
                input: 输入张量（shape: [B,C,H,W,...]）
                param: 神经元层面的缩放参数张量（必须与input同shape，逐元素对应）
                mode: 梯度计算模式（与外部模块一致）
                n: 量化等级数（与外部模块一致）
            Returns:
                output: 量化后的输出张量（与input同shape，值范围[0, n]）
            """
            # 1. 逐元素缩放：input除以param（神经元层面独立缩放）
            y = input / param  # shape与input一致：input[i,j,k,...]/param[i,j,k,...]
            # 2. 向下取整：获取整数部分
            floor_y = torch.floor(y)
            # 3. 钳位操作：将输出限制在[0, n]范围内（量化到n级）
            output = torch.clamp(floor_y, min=0, max=n)
            # 保存反向传播所需数据（张量需用save_for_backward，非张量直接存ctx）
            ctx.save_for_backward(input, param)  # 保存需要计算梯度的张量
            ctx.mode = mode  # 保存梯度模式
            ctx.n = n  # 保存量化等级数
            return output

        @staticmethod
        def backward(ctx, grad_output):
            """
            反向传播：自定义梯度计算（核心逻辑）
            Args:
                ctx: 上下文对象，获取前向保存的数据
                grad_output: 上游梯度张量（与output同shape）
            Returns:
                grad_input: input的梯度张量（与input同shape）
                grad_param: param的梯度张量（与param同shape）
            """
            # 从上下文获取前向保存的数据
            input, param = ctx.saved_tensors  # 恢复前向的input和param
            mode = ctx.mode  # 恢复梯度模式
            n_level = ctx.n  # 恢复量化等级数（避免与循环变量n冲突）

            # 校验梯度尺寸：确保上游梯度与input形状一致（逐元素梯度传播的前提）
            assert grad_output.shape == input.shape, "grad_output必须与input形状一致"

            # -------------------------- 核心：根据模式计算有效梯度掩码 --------------------------
            # 有效掩码（valid_mask）：标记input中需要保留梯度的区域（bool张量，与input同shape）
            if mode == 1:
                # 模式1：原始模式 - 计算n个离散子区间的并集
                # 生成n个等级的张量（shape: [n, 1, 1, ...]），适配input的广播机制
                k = torch.arange(1, 1 + n_level, device=input.device)  # k=1~n_level（量化等级索引）
                k = k.view(-1, *([1] * input.dim()))  # 扩展维度：[n] → [n,1,1,...]（适配input的任意维度）
                # 逐等级计算子区间：[k×param - 0.5, k×param + 0.5]（神经元层面独立计算）
                lower_bounds = k * param - 0.5  # 所有等级的下界（shape: [n, B,C,H,W,...]）
                upper_bounds = k * param + 0.5  # 所有等级的上界（shape: [n, B,C,H,W,...]）
                # 判断input是否在任意一个子区间内（沿等级维度合并结果）
                valid_mask = ((input >= lower_bounds) & (input <= upper_bounds)).any(dim=0)

            elif mode == 2:
                # 模式2：连续区间模式 - 单个连续区间（覆盖所有子区间范围，计算更快）
                lower_bound = param - 0.5  # 最低等级（k=1）的下界（逐元素）
                upper_bound = n_level * param + 0.5  # 最高等级（k=n）的上界（逐元素）
                # 判断input是否在连续区间内（直接逐元素判断，无广播开销）
                valid_mask = (input >= lower_bound) & (input <= upper_bound)

            elif mode == 3:
                # 模式3：固定区间模式 - 用1替换param，固定区间范围（最快）
                # 区间计算：param=1 → [1-0.5, n×1+0.5] = [0.5, n_level+0.5]
                lower_bound = 0.5  # 固定下界
                upper_bound = n_level + 0.5  # 固定上界（原代码笔误修正：0.5 + n_level → n_level + 0.5，逻辑一致但更清晰）
                valid_mask = (input >= lower_bound) & (input <= upper_bound)

            # -------------------------- 计算input的梯度 --------------------------
            # 安全处理param：避免除以0（防止梯度爆炸/NaN）
            safe_param = torch.clamp(torch.abs(param), min=1e-4)  # 将param绝对值限制在≥1e-4
            # 梯度公式：grad_input = 上游梯度 / param（仅有效区域保留梯度）
            grad_input = grad_output.clone() / safe_param  # 逐元素梯度缩放
            grad_input = grad_input * valid_mask.float()  # 有效区域保留梯度，无效区域置0（bool→float：True=1.0，False=0.0）

            # -------------------------- 计算param的梯度 --------------------------
            # 梯度公式：grad_param = 上游梯度 * (-input) / (param²)（仅有效区域计算）
            grad_param = torch.where(
                valid_mask,  # 逐元素判断：是否在有效区域
                grad_output * (-input) / (safe_param ** 2),  # 有效区域：应用梯度公式
                torch.tensor(0.0, device=param.device)  # 无效区域：梯度置0
            )

            # 返回input和param的梯度（顺序必须与forward的输入参数一致）
            return grad_input, grad_param, None, None

    def forward(self, x, param):
        """
        模块前向传播接口：调用自定义autograd函数
        Args:
            x: 输入张量（shape: [B,C,H,W,...]）
            param: 神经元层面缩放参数（必须与x同shape，逐元素对应）
        Returns:
            量化后的输出张量（与x同shape，值范围[0, self.n]）
        """
        # 调用自定义autograd函数的apply方法，传入所有参数（x, param, 模式, 量化等级）
        return self.quant4.apply(x, param, self.mode, self.n)

class Clip_nograd(nn.Module):
    def __init__(self, n=1):
        super(Clip_nograd, self).__init__()
        self.n = n
    class quant4(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, param, n):
            ctx.save_for_backward(input)
            ctx.param = param
            ctx.n = n
            return torch.clamp(torch.floor(input), min=0, max=n)
        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            param = ctx.param
            n = ctx.n
            grad_input = grad_output * 1
            grad_input[input < 1 - 1 / 2 / param] = 0
            grad_input[input > n + 1 / 2 / param] = 0

            return grad_input, None, None
    def forward(self, x, param):
        return self.quant4.apply(x, param, self.n)