import collections
import math
import numbers
import random

import numpy as np
import torch
import torchtoolbox.transform
import torchvision
import torchvision.transforms.functional as F
from PIL import Image, ImageOps
from torchvision import transforms


class Padding:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, imgmap):
        return [ImageOps.expand(img, border=self.pad, fill=0) for img in imgmap]


class Scale:
    def __init__(self, size, interpolation=Image.NEAREST):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgmap):
        # assert len(imgmap) > 1 # list of images
        img1 = imgmap[0]
        if isinstance(self.size, int):
            w, h = img1.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return imgmap
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return [i.resize((ow, oh), self.interpolation) for i in imgmap]
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return [i.resize((ow, oh), self.interpolation) for i in imgmap]
        else:
            return [i.resize(self.size, self.interpolation) for i in imgmap]


class CenterCrop:
    def __init__(self, size, consistent=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgmap):
        img1 = imgmap[0]
        w, h = img1.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return [i.crop((x1, y1, x1 + tw, y1 + th)) for i in imgmap]


class RandomCropWithProb:
    def __init__(self, size, p=0.8, consistent=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.consistent = consistent
        self.threshold = p

    def __call__(self, imgmap):
        img1 = imgmap[0]
        w, h = img1.size
        if self.size is not None:
            th, tw = self.size
            if w == tw and h == th:
                return imgmap
            if self.consistent:
                if random.random() < self.threshold:
                    x1 = random.randint(0, w - tw)
                    y1 = random.randint(0, h - th)
                else:
                    x1 = int(round((w - tw) / 2.))
                    y1 = int(round((h - th) / 2.))
                return [i.crop((x1, y1, x1 + tw, y1 + th)) for i in imgmap]
            else:
                result = []
                for i in imgmap:
                    if random.random() < self.threshold:
                        x1 = random.randint(0, w - tw)
                        y1 = random.randint(0, h - th)
                    else:
                        x1 = int(round((w - tw) / 2.))
                        y1 = int(round((h - th) / 2.))
                    result.append(i.crop((x1, y1, x1 + tw, y1 + th)))
                return result
        else:
            return imgmap


class RandomCrop:
    def __init__(self, size, consistent=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.consistent = consistent

    def __call__(self, imgmap, flowmap=None):
        img1 = imgmap[0]
        w, h = img1.size
        if self.size is not None:
            th, tw = self.size
            if w == tw and h == th:
                return imgmap
            if not flowmap:
                if self.consistent:
                    x1 = random.randint(0, w - tw)
                    y1 = random.randint(0, h - th)
                    return [i.crop((x1, y1, x1 + tw, y1 + th)) for i in imgmap]
                else:
                    result = []
                    for i in imgmap:
                        x1 = random.randint(0, w - tw)
                        y1 = random.randint(0, h - th)
                        result.append(i.crop((x1, y1, x1 + tw, y1 + th)))
                    return result
            elif flowmap is not None:
                assert (not self.consistent)
                result = []
                for idx, i in enumerate(imgmap):
                    proposal = []
                    for j in range(3):  # number of proposal: use the one with largest optical flow
                        x = random.randint(0, w - tw)
                        y = random.randint(0, h - th)
                        proposal.append([x, y, abs(np.mean(flowmap[idx, y:y + th, x:x + tw, :]))])
                    [x1, y1, _] = max(proposal, key=lambda x: x[-1])
                    result.append(i.crop((x1, y1, x1 + tw, y1 + th)))
                return result
            else:
                raise ValueError('wrong case')
        else:
            return imgmap


class RandomSizedCrop:
    def __init__(self, size, interpolation=Image.BILINEAR, consistent=True, p=1.0):
        self.size = size
        self.interpolation = interpolation
        self.consistent = consistent
        self.threshold = p

    def __call__(self, imgmap):
        img1 = imgmap[0]
        if random.random() < self.threshold:  # do RandomSizedCrop
            for attempt in range(10):
                area = img1.size[0] * img1.size[1]
                target_area = random.uniform(0.5, 1) * area
                aspect_ratio = random.uniform(3. / 4, 4. / 3)

                w = int(round(math.sqrt(target_area * aspect_ratio)))
                h = int(round(math.sqrt(target_area / aspect_ratio)))

                if self.consistent:
                    if random.random() < 0.5:
                        w, h = h, w
                    if w <= img1.size[0] and h <= img1.size[1]:
                        x1 = random.randint(0, img1.size[0] - w)
                        y1 = random.randint(0, img1.size[1] - h)

                        imgmap = [i.crop((x1, y1, x1 + w, y1 + h)) for i in imgmap]
                        for i in imgmap: assert (i.size == (w, h))

                        return [i.resize((self.size, self.size), self.interpolation) for i in imgmap]
                else:
                    result = []
                    for i in imgmap:
                        if random.random() < 0.5:
                            w, h = h, w
                        if w <= img1.size[0] and h <= img1.size[1]:
                            x1 = random.randint(0, img1.size[0] - w)
                            y1 = random.randint(0, img1.size[1] - h)
                            result.append(i.crop((x1, y1, x1 + w, y1 + h)))
                            assert (result[-1].size == (w, h))
                        else:
                            result.append(i)

                    assert len(result) == len(imgmap)
                    return [i.resize((self.size, self.size), self.interpolation) for i in result]

                    # Fallback
            scale = Scale(self.size, interpolation=self.interpolation)
            crop = CenterCrop(self.size)
            return crop(scale(imgmap))
        else:  # don't do RandomSizedCrop, do CenterCrop
            crop = CenterCrop(self.size)
            return crop(imgmap)


class RandomHorizontalFlip:
    def __init__(self, consistent=True, command=None):
        self.consistent = consistent
        if command == 'left':
            self.threshold = 0
        elif command == 'right':
            self.threshold = 1
        else:
            self.threshold = 0.5

    def __call__(self, imgmap):
        if self.consistent:
            if random.random() < self.threshold:
                return [i.transpose(Image.FLIP_LEFT_RIGHT) for i in imgmap]
            else:
                return imgmap
        else:
            result = []
            for i in imgmap:
                if random.random() < self.threshold:
                    result.append(i.transpose(Image.FLIP_LEFT_RIGHT))
                else:
                    result.append(i)
            assert len(result) == len(imgmap)
            return result


class RandomGray:
    '''Actually it is a channel splitting, not strictly grayscale images'''

    def __init__(self, consistent=True, p=0.5):
        self.consistent = consistent
        self.p = p  # probability to apply grayscale

    def __call__(self, imgmap):
        if self.consistent:
            if random.random() < self.p:
                return [self.grayscale(i) for i in imgmap]
            else:
                return imgmap
        else:
            result = []
            for i in imgmap:
                if random.random() < self.p:
                    result.append(self.grayscale(i))
                else:
                    result.append(i)
            assert len(result) == len(imgmap)
            return result

    def grayscale(self, img):
        channel = np.random.choice(3)
        np_img = np.array(img)[:, :, channel]
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, 'RGB')
        return img


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image. --modified from pytorch source code
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, consistent=False, p=1.0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        self.consistent = consistent
        self.threshold = p

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = torchvision.transforms.Compose(transforms)

        return transform

    def __call__(self, imgmap):
        if random.random() < self.threshold:  # do ColorJitter
            if self.consistent:
                transform = self.get_params(self.brightness, self.contrast,
                                            self.saturation, self.hue)
                return [transform(i) for i in imgmap]
            else:
                result = []
                for img in imgmap:
                    transform = self.get_params(self.brightness, self.contrast,
                                                self.saturation, self.hue)
                    result.append(transform(img))
                return result
        else:  # don't do ColorJitter, do nothing
            return imgmap

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


class RandomRotation:
    def __init__(self, consistent=True, degree=15, p=1.0):
        self.consistent = consistent
        self.degree = degree
        self.threshold = p

    def __call__(self, imgmap):
        if random.random() < self.threshold:  # do RandomRotation
            if self.consistent:
                deg = np.random.randint(-self.degree, self.degree, 1)[0]
                return [i.rotate(deg, expand=True) for i in imgmap]
            else:
                return [i.rotate(np.random.randint(-self.degree, self.degree, 1)[0], expand=True) for i in imgmap]
        else:  # don't do RandomRotation, do nothing
            return imgmap


class ToTensor:
    def __call__(self, imgmap):
        totensor = transforms.ToTensor()
        return [totensor(i) for i in imgmap]


class ToPILImage:
    def __call__(self, imgmap):
        topilimage = transforms.ToPILImage()
        return [topilimage(i) for i in imgmap]


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, imgmap):
        resize = transforms.Resize(self.size)
        return [resize(i) for i in imgmap]


class Cutout:
    def __call__(self, imgmap):
        # default 0.5
        if random.random() < 0.5:
            cutout = torchtoolbox.transform.Cutout(p=1)
            return [cutout(i) for i in imgmap]
        else:
            return imgmap


class Normalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, imgmap):
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        return [normalize(i) for i in imgmap]


class Roll:
    def __init__(self):
        self.off1 = random.randint(-5, 5)
        self.off2 = random.randint(-5, 5)

    def __call__(self, imgmap):
        return [torch.roll(i, shifts=(self.off1, self.off2), dims=(1, 2)) for i in imgmap]


import math
from typing import Dict, Tuple, Optional, List
import torch
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import (
    affine, rotate, adjust_brightness, adjust_saturation,
    adjust_contrast, adjust_sharpness, posterize, solarize,
    autocontrast, equalize, invert
)
from torchvision.transforms import RandomErasing


class SNNAugmentWide(torch.nn.Module):
    r"""适配帧序列的TrivialAugment Wide增强，支持对多帧数据中的每帧单独应用随机增强。
    输入应为帧序列（列表或Tensor，形状为(T, C, H, W)），输出同类型的增强后序列。
    """

    def __init__(self, num_magnitude_bins: int = 31,
                 interpolation: InterpolationMode = InterpolationMode.NEAREST,
                 fill: Optional[List[float]] = None) -> None:
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill
        # 初始化Cutout（随机擦除），概率设为1（因为会在op中随机选择是否应用）
        self.cutout = RandomErasing(p=1, scale=(0.05, 0.33), ratio=(0.2, 5))

    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # 仅保留几何变换（与通道数无关，支持2通道）
            "Identity": (torch.tensor(0.0), False),  # 无增强
            "ShearX": (torch.linspace(-0.3, 0.3, num_bins), True),  # X方向剪切
            # "ShearY": (torch.linspace(-0.3, 0.3, num_bins), True),  # Y方向剪切
            "TranslateX": (torch.linspace(-5.0, 5.0, num_bins), True),  # X方向平移
            "TranslateY": (torch.linspace(-5.0, 5.0, num_bins), True),  # Y方向平移
            "Rotate": (torch.linspace(-30.0, 30.0, num_bins), True),  # 旋转
            "Cutout": (torch.linspace(1.0, 30.0, num_bins), True),  # 随机擦除（支持任意通道）
        }

    def forward(self, img_seq):
        """
        Args:
            img_seq: 帧序列，支持两种格式：
                - 列表：每个元素为单帧Tensor，形状为(C, H, W)
                - Tensor：形状为(T, C, H, W)（T为时间步，即帧数量）

        Returns:
            增强后的帧序列，格式与输入一致
        """
        # 统一转换为列表格式处理（方便遍历）
        if isinstance(img_seq, Tensor):
            is_tensor = True
            img_list = [img_seq[t] for t in range(img_seq.shape[0])]  # 拆分为单帧列表
        else:
            is_tensor = False
            img_list = img_seq  # 已为列表

        # 为当前批次生成一组随机增强参数（所有帧共享同一套参数，保持时序一致性）
        op_meta = self._augmentation_space(self.num_magnitude_bins)
        op_index = int(torch.randint(len(op_meta), (1,)).item())
        op_name = list(op_meta.keys())[op_index]
        magnitudes, signed = op_meta[op_name]
        magnitude = float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item()) \
            if magnitudes.ndim > 0 else 0.0
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0

        # 对每帧应用相同的增强操作（确保时序一致性）
        augmented_list = []
        for img in img_list:
            # 处理填充值（适配单帧通道数）
            fill = self.fill
            if isinstance(img, Tensor):
                if isinstance(fill, (int, float)):
                    fill = [float(fill)] * img.shape[0]  # 按通道数复制填充值
                elif fill is not None:
                    fill = [float(f) for f in fill]

            # 应用增强操作
            if op_name == "Cutout":
                augmented_img = self.cutout(img)
            else:
                augmented_img = self._apply_op(img, op_name, magnitude, self.interpolation, fill)
            augmented_list.append(augmented_img)

        # 还原为输入格式（Tensor或列表）
        if is_tensor:
            return torch.stack(augmented_list, dim=0)  # 重组为(T, C, H, W)
        else:
            return augmented_list

    def _apply_op(self, img: Tensor, op_name: str, magnitude: float,
                  interpolation: InterpolationMode, fill: Optional[List[float]]):
        """对单帧应用增强操作"""
        if op_name == "ShearX":
            return affine(img, angle=0.0, translate=[0, 0], scale=1.0,
                          shear=[math.degrees(magnitude), 0.0],  # X方向剪切角度
                          interpolation=interpolation, fill=fill)
        elif op_name == "ShearY":
            return affine(img, angle=0.0, translate=[0, 0], scale=1.0,
                          shear=[0.0, math.degrees(magnitude)],  # Y方向剪切角度
                          interpolation=interpolation, fill=fill)
        elif op_name == "TranslateX":
            return affine(img, angle=0.0, translate=[int(magnitude), 0], scale=1.0,
                          interpolation=interpolation, shear=[0.0, 0.0], fill=fill)
        elif op_name == "TranslateY":
            return affine(img, angle=0.0, translate=[0, int(magnitude)], scale=1.0,
                          interpolation=interpolation, shear=[0.0, 0.0], fill=fill)
        elif op_name == "Rotate":
            return rotate(img, magnitude, interpolation=interpolation, fill=fill)
        elif op_name == "Brightness":
            return adjust_brightness(img, 1.0 + magnitude)  # 1.0为基准，magnitude可正负
        elif op_name == "Contrast":
            return adjust_contrast(img, 1.0 + magnitude)
        elif op_name == "Sharpness":
            return adjust_sharpness(img, 1.0 + magnitude)
        elif op_name == "Identity":
            return img  # 无增强
        else:
            raise ValueError(f"不支持的增强操作: {op_name}")

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += f'num_magnitude_bins={self.num_magnitude_bins}, '
        s += f'interpolation={self.interpolation}, '
        s += f'fill={self.fill}'
        s += ')'
        return s




if __name__ == '__main__':
    from utils.cifar10_dvs import CIFAR10DVS

    c_in = 2
    num_classes = 10

    transform_train = transforms.Compose([
        # CIFAR10_DVS_Aug(), # it has been resize 48
        ToPILImage(),
        Resize(48),
        RandomSizedCrop(48),
        RandomHorizontalFlip(),
        ToTensor(),
        # Roll(),
    ])

    transform_test = transforms.Compose([
        ToPILImage(),
        Resize(48),
        ToTensor(),
    ])
    data_dir = "./data_dir"
    trainset = CIFAR10DVS(data_dir, train=True, use_frame=True, frames_num=10, split_by='number',
                          normalization=None, transform=transform_train)
    testset = CIFAR10DVS(data_dir, train=False, use_frame=True, frames_num=10, split_by='number',
                         normalization=None, transform=transform_test)

    # train_data_loader = data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
    # test_data_loader = data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)
    print(trainset[0])
