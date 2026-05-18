import math

import numpy as np

from spikingjelly.clock_driven import surrogate
import torch
import torch.nn as nn
import torch.nn.functional as F
Tensor = torch.Tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)

class Layer(nn.Module):  # baseline
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding):
        super(Layer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
            nn.BatchNorm2d(out_plane)
        )
        # self.act = LIFSpike()

    def forward(self, x):
        x = self.fwd(x)
        # x = self.act(x)
        return x

class TEBN(nn.Module):
    def __init__(self, out_plane, eps=1e-5, momentum=0.1):
        super(TEBN, self).__init__()
        self.bn = SeqToANNContainer(nn.BatchNorm2d(out_plane))
        # p ~ T
        self.p = nn.Parameter(torch.ones(8, 1, 1, 1, 1, device=device))
    def forward(self, input):
        y = self.bn(input)
        y = y.transpose(0, 1).contiguous()  # NTCHW  TNCHW
        y = y * self.p
        y = y.contiguous().transpose(0, 1)  # TNCHW  NTCHW
        return y

class TEBNLayer(nn.Module):  # baseline+TN
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding):
        super(TEBNLayer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
        )
        self.bn = TEBN(out_plane)
        # self.act = LIFSpike()

    def forward(self, x):
        y = self.fwd(x)
        y = self.bn(y)
        # x = self.act(x)
        return y



class ZIF(torch.autograd.Function):
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

# add
class aLIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.25, gamma=1.0):
        super(aLIFSpike, self).__init__()
        # self.heaviside = ZIF.apply
        self.heaviside = quant4.apply
        self.v_th = thresh
        self.tau = tau
        self.a = nn.Parameter(torch.tensor(.0), requires_grad=True)
        self.c = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.gamma = gamma

    def forward(self, x):
        mem_v = []

        mem = 0

        N, T, C, H, W = x.shape

        m = torch.zeros_like(x)

        for t in range(T):
            mem = self.tau * mem + x[:, t, ...]
            m[:, t, ...] = m[:, t-1, ...].relu() * torch.sigmoid(self.c * x[:, t, ...]) + -(-m[:, t-1, ...]).relu() * (1 - torch.sigmoid(self.c * x[:, t, ...]))
            # spike = self.heaviside(mem - 1 - self.a.tanh() * x[:, t, ...].tanh(), self.gamma)
            spike = self.heaviside(mem, 1 + self.a.tanh() * x[:, t, ...].tanh())
            m[:, t, ...] += spike * torch.sigmoid(x[:, t, ...])
            m[:, t, ...] -= (1 - spike) * torch.sigmoid(x[:, t, ...])
            mem = mem - spike * (1 + torch.sigmoid(m[:, t, ...]) + self.a.tanh() * x[:, t, ...].tanh())
            mem_v.append(spike)

        return torch.stack(mem_v, dim=1)
# add

# add
class bLIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.25, gamma=1.0):
        super(bLIFSpike, self).__init__()
        # self.heaviside = ZIF.apply
        self.heaviside = quant4.apply
        self.v_th = thresh
        self.tau = tau
        self.a = nn.Parameter(torch.tensor(.0), requires_grad=True)
        self.grad_h = nn.Parameter(torch.tensor(.0), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.d = nn.Parameter(torch.tensor(.0), requires_grad=True)

    def forward(self, x):
        mem_v = []

        mem = 0

        N, T, C, H, W = x.shape

        m = torch.zeros_like(x[:, 0, ...])

        for t in range(T):
            mem = self.tau * mem + x[:, t, ...]
            spike = self.heaviside(mem, 1 + self.a.tanh() * x[:, t, ...].tanh())
            sign_nograd = torch.sign(spike)
            sign_withgrad = self.grad_h.sigmoid() * spike + (sign_nograd - self.grad_h.sigmoid() * spike).detach()
            # reset voltage leaky
            m = m * torch.sigmoid(self.d)
            # reset voltage integrate
            m = m + sign_withgrad * torch.sigmoid(self.b * x[:, t, ...]) - (1 - sign_withgrad) * (1 - torch.sigmoid(self.b * x[:, t, ...]))
            # ad reset
            mem = mem - spike * (1 + self.a.tanh() * x[:, t, ...].tanh()) - sign_withgrad * torch.sigmoid(m)
            mem_v.append(spike)

        return torch.stack(mem_v, dim=1)
# add

class LIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.25, gamma=1.0):
        super(LIFSpike, self).__init__()
        self.heaviside = ZIF.apply
        self.v_th = thresh
        self.tau = tau
        self.gamma = gamma
        self.pre_spike_mem = []

    def forward(self, x):
        mem_v = []
        # _mem = []
        mem = 0
        T = x.shape[1]
        for t in range(T):
            mem = self.tau * mem + x[:, t, ...]
            # _mem.append(mem.detach().cpu().clone())
            spike = self.heaviside(mem - self.v_th, self.gamma)
            mem = mem * 1 - spike
            mem_v.append(spike)
        # self.pre_spike_mem = torch.stack(_mem)
        return torch.stack(mem_v, dim=1)

class MaskedSlidingPSN(nn.Module):

    def gen_gemm_weight(self, T: int):
        weight = torch.zeros([T, T], device=self.weight.device)
        for i in range(T):
            end = i + 1
            start = max(0, i + 1 - self.order)
            length = min(end - start, self.order)
            weight[i][start: end] = self.weight[self.order - length: self.order]

        return weight


    def __init__(self, order: int = 2, surrogate_function = surrogate.ATan(), exp_init: bool=True):
        super().__init__()

        self.order = order
        if exp_init:
            weight = torch.ones([order])
            for i in range(order - 2, -1, -1):
                weight[i] = weight[i + 1] / 2.

            self.weight = nn.Parameter(weight)
        else:
            self.weight = torch.ones([1, order])
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            self.weight = nn.Parameter(self.weight[0])
        self.threshold = nn.Parameter(torch.as_tensor(-1.))
        self.surrogate_function = surrogate_function


    def forward(self, x_seq: torch.Tensor):
        # x_seq.shape = [N, T, *]
        weight = self.gen_gemm_weight(x_seq.shape[1])
        h_seq = F.linear(x_seq.transpose(1, -1), weight, self.threshold)
        h_seq = h_seq.transpose(1, -1)

        return self.surrogate_function(h_seq)

class VGGPSN(nn.Module):
    def __init__(self, tau=0.5):
        super(VGGPSN, self).__init__()
        self.tau = tau
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        # pool = APLayer(2)
        self.features = nn.Sequential(
            Layer(2, 64, 3, 1, 1),
            MaskedSlidingPSN(),
            Layer(64, 128, 3, 1, 1),
            MaskedSlidingPSN(),
            pool,
            Layer(128, 256, 3, 1, 1),
            MaskedSlidingPSN(),
            Layer(256, 256, 3, 1, 1),
            MaskedSlidingPSN(),
            pool,
            Layer(256, 512, 3, 1, 1),
            MaskedSlidingPSN(),
            Layer(512, 512, 3, 1, 1),
            MaskedSlidingPSN(),
            pool,
            Layer(512, 512, 3, 1, 1),
            MaskedSlidingPSN(),
            Layer(512, 512, 3, 1, 1),
            MaskedSlidingPSN(),
            pool,
        )
        W = int(48 / 2 / 2 / 2 / 2)
        # self.T = 10
        self.classifier = nn.Sequential(SeqToANNContainer(nn.Dropout2d(0.25), nn.Linear(512 * W * W, 10)))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # print(input.shape)
        # input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x

class VGGSNN(nn.Module):
    def __init__(self, tau=0.5):
        super(VGGSNN, self).__init__()
        self.tau = tau
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        # pool = APLayer(2)
        self.features = nn.Sequential(
            Layer(2, 64, 3, 1, 1),
            bLIFSpike(tau=self.tau),
            Layer(64, 128, 3, 1, 1),
            bLIFSpike(tau=self.tau),
            pool,
            Layer(128, 256, 3, 1, 1),
            bLIFSpike(tau=self.tau),
            Layer(256, 256, 3, 1, 1),
            bLIFSpike(tau=self.tau),
            pool,
            Layer(256, 512, 3, 1, 1),
            bLIFSpike(tau=self.tau),
            Layer(512, 512, 3, 1, 1),
            bLIFSpike(tau=self.tau),
            pool,
            Layer(512, 512, 3, 1, 1),
            bLIFSpike(tau=self.tau),
            Layer(512, 512, 3, 1, 1),
            bLIFSpike(tau=self.tau),
            pool,
        )
        W = int(48 / 2 / 2 / 2 / 2)
        # self.T = 10
        self.classifier = nn.Sequential(nn.Dropout(0.25), SeqToANNContainer(nn.Linear(512 * W * W, 10)))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x




def MH(x):
    return (0.75 - x.pow(2)) / (0.75 * 0.75 * math.sqrt(4.71225)) * torch.exp(-x.pow(2) / 1.5)

def Swish(x):
    return x / (1 + torch.exp(-0.5 * x))

class rectangle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, vth):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.vth = vth
        return surrogate.heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            x = ctx.saved_tensors[0]
            mask1 = (x.abs() > ctx.vth / 2)
            mask_ = mask1.logical_not()
            grad_x = grad_output * x.masked_fill(mask_, 1. / ctx.vth).masked_fill(mask1, 0.)
        return grad_x, None

class quant4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, param):
        # 前向计算：逐元素除法 → 取整 → 钳位（同尺寸操作）
        y = input / param  # 逐元素: input[i,j,k,l]/param[i,j,k,l]
        floor_y = torch.floor(y)
        output = torch.clamp(floor_y, min=0, max=1)
        # 保存反向所需的张量（均为同尺寸）
        ctx.save_for_backward(input, param)
        ctx.floor_y = floor_y
        return output
    @staticmethod
    def backward(ctx, grad_output):
        input, param = ctx.saved_tensors
        # TEST benshen xiaoguo
        # lower_bounds = 1 * param - 0.5
        # upper_bounds = 16 * param + 0.5
        # valid_mask = ((input >= lower_bounds) & (input <= upper_bounds))
        # grad_output与input同尺寸（反向传播的梯度尺寸匹配）
        assert grad_output.shape == input.shape, "grad_output must match input shape"
        # 生成n=1~16，形状为 [16, 1, 1, 1, 1]，确保与input/param的所有维度可广播
        # 维度数量 = input维度 + 1（n的维度），且n的非1维度在最前面
        n = torch.arange(1, 1 + 1, device=input.device)  # 形状 [16]
        # 扩展n的维度，使其与input的每个维度都有一个对应1的维度
        n = n.view(-1, *([1] * input.dim()))  # 对于4D input，形状变为 [16, 1, 1, 1, 1]
        # 无需扩展input和param，直接利用广播机制 # input形状 [B, C, H, W] 会自动广播为 [16, B, C, H, W] # param同理
        lower_bounds = n * param - 0.5
        upper_bounds = n * param + 0.5
        # 计算掩码：input在任何n对应的区间内即有效 # 结果形状 [16, B, C, H, W]，沿n的维度合并
        valid_mask = ((input >= lower_bounds) & (input <= upper_bounds)).any(dim=0)  # 合并后形状 [B, C, H, W]
        # 验证掩码维度是否与input一致（关键检查）
        assert valid_mask.shape == input.shape, f"掩码维度不匹配: {valid_mask.shape} vs {input.shape}"
        # 1. grad_input：逐元素矩形替代梯度（与原逻辑一致，适配同尺寸param）
        # 逐元素计算有效区域：input ∈ [param-0.5, param+0.5]
        # valid_mask = (input >= (1 * param - 0.5)) & (input <= (16 * param + 0.5))  # 同尺寸bool张量
        # 2025/10/13 debug：缺少 1 / vth！
        grad_input = grad_output.clone()
        grad_input = grad_input * valid_mask.float()  # 有效区域保留梯度，无效区域置0
        # 2. grad_param：逐元素梯度计算（适配同尺寸param）
        # 仅在有效区域计算梯度，公式：grad_output * (-input) / (param²)
        grad_param = torch.where(
            valid_mask,  # 逐元素判断有效区域
            grad_output * (-input) / (param ** 2),  # 逐元素梯度公式
            torch.tensor(0.0, device=param.device)  # 无效区域梯度为0
        )
        # grad_param与param同尺寸，无需聚合（逐元素对应更新）
        return grad_input, grad_param