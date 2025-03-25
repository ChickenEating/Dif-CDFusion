
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
import torchvision.transforms.functional as TF
import kornia
import cv2



class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return k


class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_fused, thresholds):
        image_A_Y = image_A[:, :1, :, :]
        image_B_Y = image_B[:, :1, :, :]
        image_fused_Y = image_fused[:, :1, :, :]
        gradient_A = self.sobelconv(image_A_Y)
        # gradient_A = TF.gaussian_blur(gradient_A, 3, [1, 1])
        gradient_B = self.sobelconv(image_B_Y)
        # gradient_B = TF.gaussian_blur(gradient_B, 3, [1, 1])
        gradient_fused = self.sobelconv(image_fused_Y)
        # gradient_joint = torch.max(gradient_A, gradient_B)
        grant_joint = torch.concat([gradient_A, gradient_B], dim=1)
        grant_joint_max, index = grant_joint.max(dim=1)

        a, b, c, d = gradient_A.size(0), gradient_A.size(1), gradient_A.size(2), gradient_A.size(3)
        grant_joint_max = grant_joint_max.reshape(a, b, c, d)

        gradient_A_Mask = threshold_tensor(gradient_A, dim=2, k=thresholds)
        aaa = gradient_A_Mask.argmax(dim=1).shape
        gradient_B_Mask = threshold_tensor(gradient_B, dim=2, k=thresholds)
        bbb = gradient_B_Mask.argmax(dim=1).shape



        Loss_gradient = F.l1_loss(gradient_fused, grant_joint_max)
        return Loss_gradient, gradient_A_Mask, gradient_B_Mask



def gradWeightBlockIntenLoss(image_A_Y, image_B_Y, image_fused_Y, gradient_A, gradient_B, L_Inten_loss, percent, mask_pre = None):
    """
    percent:百分比，大于百分之多少的像素点
    L_Inten_loss: 计算像素损失的函数
    gradient_A:A图像的梯度
    mask_pre:前一次的掩膜，第一次前百分之20，第二次取60，就是中间的四十
    """
    thresholds = torch.round(torch.tensor(percent * image_A_Y.shape[2] * image_A_Y.shape[3])).int()
    clone_grand_A = gradient_A.clone().detach()
    gradient_A_Mask = threshold_tensor(clone_grand_A, dim=2, k=thresholds)


    clone_grand_B = gradient_B.clone().detach()
    gradient_B_Mask = threshold_tensor(clone_grand_B, dim=2, k=thresholds)

    if mask_pre == None:
        grand_Mask = gradient_A_Mask + gradient_B_Mask
        grand_Mask = grand_Mask.clamp(min=0, max=1)

    else:
        grand_Mask = gradient_A_Mask + gradient_B_Mask
        grand_Mask = grand_Mask.clamp(min=0, max=1)

        grand_Mask -= mask_pre
    grand_IntenLoss = L_Inten_loss(image_A_Y * grand_Mask, image_B_Y * grand_Mask, image_fused_Y * grand_Mask)
    return grand_IntenLoss, grand_Mask


def testNum(grand_Mask):
    grand_Mask_1Wei = torch.flatten(grand_Mask)
    num = 0
    for i in range(grand_Mask_1Wei.shape[0]):
        if grand_Mask_1Wei[i] == 1:
            num += 1
    return num

# class L_Grad_Inte(nn.Module):
    """
        按梯度分块求像素损失并计算梯度损失
    """
    def __init__(self):
        super(L_Grad_Inte, self).__init__()
        self.sobelconv=Sobelxy()
        self.L_Inten_aver = L_IntensityAver()
        self.L_Inten_Max = L_Intensity()
        self.L_Inten_Once = L_IntensityOnce()
    def forward(self, image_A, image_B, image_fused):
        # image_A_Y = image_A[:, :1, :, :]
        # image_B_Y = image_B[:, :1, :, :]
        # image_fused_Y = image_fused[:, :1, :, :]
        image_A_Y = image_A[:, :, :, :]
        image_B_Y = image_B[:, :, :, :]
        image_fused_Y = image_fused[:, :, :, :]
        
        gradient_A = self.sobelconv(image_A_Y)
        gradient_B = self.sobelconv(image_B_Y)
        gradient_fused = self.sobelconv(image_fused_Y)
        grant_joint = torch.cat([gradient_A, gradient_B], dim=1)
        grant_joint_max, index = grant_joint.max(dim=1)

        a, b, c, d = gradient_A.size(0), gradient_A.size(1), gradient_A.size(2), gradient_A.size(3)
        grant_joint_max = grant_joint_max.reshape(a, b, c, d)

#梯度乘以像素强度来对图像进行分等级求强度loss
        gradient_A_Att = image_A_Y * gradient_A
        gradient_B_Att = image_B_Y * gradient_B


#前百分之20的梯度的像素点用max像素损失
        grand_IntenLoss_one, grand_Mask_one = gradWeightBlockIntenLoss(image_A_Y, image_B_Y, image_fused_Y, gradient_A_Att, gradient_B_Att, self.L_Inten_Max, 0.8, mask_pre = None)
# #百分之20-70的用平均
#         grand_IntenLoss_two, grand_Mask_two = gradWeightBlockIntenLoss(image_A_Y, image_B_Y, image_fused_Y, gradient_A_Att, gradient_B_Att, self.L_Inten_aver, 0.3, mask_pre = grand_Mask_one)
# 最后30用vi的像素点
        grand_Mask_three = 1 - grand_Mask_one
        grand_IntenLoss_three = self.L_Inten_aver(image_A_Y * grand_Mask_three, image_B_Y * grand_Mask_three, image_fused_Y * grand_Mask_three)

        grand_IntenLoss = grand_IntenLoss_one + grand_IntenLoss_three

        Loss_gradient = F.l1_loss(gradient_fused, grant_joint_max)
        return Loss_gradient, grand_IntenLoss

# class L_Grad_Inte(nn.Module):
    """
        按梯度分块求像素损失并计算梯度损失，支持多通道
    """
    def __init__(self):
        super(L_Grad_Inte, self).__init__()
        self.sobelconv = Sobelxy()
        self.L_Inten_aver = L_IntensityAver()
        self.L_Inten_Max = L_Intensity()
        self.L_Inten_Once = L_IntensityOnce()

    def forward(self, image_A, image_B, image_fused):
        # 初始化损失
        total_gradient_loss = 0
        total_intensity_loss = 0
        
        # 遍历每个通道（对于RGB，遍历0, 1, 2）
        for c in range(image_A.size(1)):  # image_A.size(1)是通道数 (通常是3)
            # 提取当前通道
            image_A_channel = image_A[:, c:c+1, :, :]
            image_B_channel = image_B[:, c:c+1, :, :]
            image_fused_channel = image_fused[:, c:c+1, :, :]

            # 计算梯度
            gradient_A = self.sobelconv(image_A_channel)
            gradient_B = self.sobelconv(image_B_channel)
            gradient_fused = self.sobelconv(image_fused_channel)

            # 梯度拼接
            grant_joint = torch.cat([gradient_A, gradient_B], dim=1)
            grant_joint_max, _ = grant_joint.max(dim=1)

            # 计算梯度与强度的加权损失
            gradient_A_Att = image_A_channel * gradient_A
            gradient_B_Att = image_B_channel * gradient_B

            # 前80%使用最大像素损失
            grand_IntenLoss_one, grand_Mask_one = gradWeightBlockIntenLoss(
                image_A_channel, image_B_channel, image_fused_channel,
                gradient_A_Att, gradient_B_Att, self.L_Inten_Max, 0.8
            )

            # 后30%使用均值强度损失
            grand_Mask_three = 1 - grand_Mask_one
            grand_IntenLoss_three = self.L_Inten_aver(
                image_A_channel * grand_Mask_three,
                image_B_channel * grand_Mask_three,
                image_fused_channel * grand_Mask_three
            )

            # 总强度损失
            grand_IntenLoss = grand_IntenLoss_one + grand_IntenLoss_three

            # 梯度损失
            Loss_gradient = F.l1_loss(gradient_fused.squeeze(1), grant_joint_max)

            total_gradient_loss += Loss_gradient
            total_intensity_loss += grand_IntenLoss

        # 返回总的梯度损失和强度损失
        return total_gradient_loss, total_intensity_loss


def rgb_to_ycbcr(image):
    """
    Convert an RGB image (PyTorch tensor) to YCbCr space.
    """
    device = image.device
    image = image.permute(0, 2, 3, 1).detach().cpu().numpy()  # 转换为 NHWC 格式
    ycbcr = []
    for img in image:
        ycbcr.append(cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2YCrCb))
    ycbcr = torch.from_numpy(np.stack(ycbcr) / 255.0).permute(0, 3, 1, 2).to(device)  # 转回 NCHW
    return ycbcr

def extract_cb_cr(ycbcr_image):
    """
    Extract the Cb and Cr channels from YCbCr image.
    """
    cb = ycbcr_image[:, 1:2, :, :]  # Cb 通道
    cr = ycbcr_image[:, 2:3, :, :]  # Cr 通道
    return cb, cr

def extract_a_b(lab_image):
    """
    Extract the a and b channels from Lab image.
    """
    a = lab_image[:, 1:2, :, :]  # a 通道
    b = lab_image[:, 2:3, :, :]  # b 通道
    return a, b

def rgb_to_lab(image):
    """
    Convert an RGB image (PyTorch tensor, normalized to [0, 1]) to Lab color space.
    :param image: Tensor of shape (batch_size, 3, height, width), RGB channels normalized to [0, 1].
    :return: Tensor of shape (batch_size, 3, height, width), Lab channels normalized to standard ranges.
    """
    assert image.size(1) == 3, "Input image must have 3 channels (RGB)."

    # Convert RGB to linear RGB
    def srgb_to_linear(rgb):
        linear = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        return linear

    # Convert linear RGB to XYZ
    def linear_rgb_to_xyz(linear_rgb):
        r, g, b = linear_rgb[:, 0:1, :, :], linear_rgb[:, 1:2, :, :], linear_rgb[:, 2:3, :, :]
        X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
        Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
        Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
        return torch.cat([X, Y, Z], dim=1)

    # Normalize XYZ using D65 white point
    def normalize_xyz(xyz):
        ref_X, ref_Y, ref_Z = 0.95047, 1.00000, 1.08883  # D65 illuminant
        X, Y, Z = xyz[:, 0:1, :, :], xyz[:, 1:2, :, :], xyz[:, 2:3, :, :]
        X, Y, Z = X / ref_X, Y / ref_Y, Z / ref_Z
        return torch.cat([X, Y, Z], dim=1)

    # Convert normalized XYZ to Lab
    def xyz_to_lab(xyz):
        def f(t):
            delta = 6 / 29
            return torch.where(t > delta ** 3, t ** (1 / 3), (t / (3 * delta ** 2)) + (4 / 29))

        X, Y, Z = xyz[:, 0:1, :, :], xyz[:, 1:2, :, :], xyz[:, 2:3, :, :]
        fX, fY, fZ = f(X), f(Y), f(Z)
        L = (116 * fY) - 16
        a = 500 * (fX - fY)
        b = 200 * (fY - fZ)
        return torch.cat([L, a, b], dim=1)

    # Step 1: Convert sRGB to linear RGB
    linear_rgb = srgb_to_linear(image)

    # Step 2: Convert linear RGB to XYZ
    xyz = linear_rgb_to_xyz(linear_rgb)

    # Step 3: Normalize XYZ
    normalized_xyz = normalize_xyz(xyz)

    # Step 4: Convert normalized XYZ to Lab
    lab = xyz_to_lab(normalized_xyz)

    return lab



class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy()
        self.vgg = vgg16(pretrained=True).features[:8].eval()  # 使用VGG的感知损失部分
        for param in self.vgg.parameters():
            param.requires_grad = False

    def normalize(self, image):
        """
        Normalize the image to [0, 1] if its maximum value is > 1.
        """
        if image.max() > 1:
            return image / 255.0
        return image
    
    def perceptual_loss(self, fused, visible):
        fused = self.normalize(fused)
        visible = self.normalize(visible)
        fused_features = self.vgg(fused)
        visible_features = self.vgg(visible)
        return F.l1_loss(fused_features, visible_features)
    
    def color_loss_ycbcr(self, fused, visible):
        # 转换到 YCbCr 空间
        fused_ycbcr = rgb_to_ycbcr(fused)
        visible_ycbcr = rgb_to_ycbcr(visible)

        # 提取 Cb 和 Cr 通道
        fused_cb, fused_cr = extract_cb_cr(fused_ycbcr)
        visible_cb, visible_cr = extract_cb_cr(visible_ycbcr)

        # 计算 Cb 和 Cr 的 L1 损失
        cb_loss = F.l1_loss(fused_cb, visible_cb)
        cr_loss = F.l1_loss(fused_cr, visible_cr)
        return cb_loss + cr_loss

    def color_loss_lab(self, fused, visible):
        visible = self.normalize(visible)
        fused = self.normalize(fused)
        # 转换到 Lab 空间
        fused_lab = rgb_to_lab(fused)
        visible_lab = rgb_to_lab(visible)

        # 提取 a 和 b 通道
        fused_a, fused_b = extract_a_b(fused_lab)
        visible_a, visible_b = extract_a_b(visible_lab)

        # 计算 a 和 b 的 L1 损失
        a_loss = F.l1_loss(fused_a, visible_a)
        b_loss = F.l1_loss(fused_b, visible_b)
        return a_loss + b_loss
    
    def forward(self, image_vis, image_ir, generate_img):
        # 归一化
        image_vis = self.normalize(image_vis)
        image_ir = self.normalize(image_ir)
        generate_img = self.normalize(generate_img)
        
        # 强度损失
        image_y = image_vis
        B, C, W, H = image_vis.shape
        image_ir = image_ir.expand(B, C, W, H)
        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(generate_img, x_in_max)
        
        # 梯度损失
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        B, C, K, W, H = y_grad.shape
        ir_grad = ir_grad.expand(B, C, K, W, H)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.maximum(y_grad, ir_grad)
        loss_grad = F.l1_loss(generate_img_grad, x_grad_joint)
        
        # 感知损失
        perceptual_loss = self.perceptual_loss(generate_img, image_vis)
        
        # 颜色空间损失（使用 YCbCr 或 Lab 二选一）
        color_loss = self.color_loss_ycbcr(generate_img, image_vis)
        # 如果想用 Lab，可以替换成：
        # color_loss = self.color_loss_lab(generate_img, image_vis)
        
        return loss_grad, loss_in, perceptual_loss, color_loss


def threshold_tensor(input_tensor, dim, k):
    """
    将输入的Tensor按维度dim取第k大的元素作为阈值，大于等于阈值的元素置为1，其余元素置为0。

    Args:
    - input_tensor: 输入的Tensor
    - dim: 取第k大元素的维度
    - k: 取第k大元素

    Returns:
    - 输出的Tensor，形状与输入的Tensor相同
    """
    # kth_value, _ = torch.kthvalue(input_tensor, k, dim=dim, keepdim=True)  # 按维度dim取第k大的元素
    B, N, C ,D = input_tensor.shape
    input_tensor = input_tensor.reshape(B,N,C*D)
    for i in range(B):
        kth_value, _ = torch.kthvalue(input_tensor[i:i+1, :, :], k, dim=dim, keepdim=True)
        kth_value = torch.flatten(kth_value)
        input_tensor[i:i+1,: , :] = torch.where(input_tensor[i:i+1, :, :] >= kth_value[0], torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())
    input_tensor = input_tensor.reshape(B, N, C ,D)
    return input_tensor


# class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        b, c, w, h = x.shape
        batch_list = []
        for i in range(b):
            tensor_list = []
            for j in range(c):
                sobelx_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weightx, padding=1)
                sobely_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weighty, padding=1)
                add_0 = torch.abs(sobelx_0) + torch.abs(sobely_0)
                tensor_list.append(add_0)

            batch_list.append(torch.stack(tensor_list, dim=1))

        return torch.cat(batch_list, dim=0)

class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_A, image_B, image_fused):
        intensity_joint = torch.max(image_A, image_B)
        Loss_intensity = F.l1_loss(intensity_joint, image_fused)
        return Loss_intensity


class L_IntensityAver(nn.Module):
    def __init__(self):
        super(L_IntensityAver, self).__init__()

    def forward(self, image_A, image_B, image_fused):
        Loss_intensity_A = F.l1_loss(image_A, image_fused)
        Loss_intensity_B = F.l1_loss(image_B, image_fused)
        Loss_intensity = 0.5 * Loss_intensity_A + 0.5 * Loss_intensity_B
        return Loss_intensity


class L_IntensityOnce(nn.Module):
    def __init__(self):
        super(L_IntensityOnce, self).__init__()

    def forward(self, image_A, image_fused):

        Loss_intensity = F.l1_loss(image_A, image_fused)
        return Loss_intensity


class L_Intensity_GrandFu(nn.Module):
    def __init__(self):
        super(L_Intensity_GrandFu, self).__init__()

    def forward(self,image_A, image_B, image_fused, gradient_A_Mask, gradient_B_Mask):

        Fu_image_maskA_A = image_A * gradient_A_Mask
        Loss_intensity_maskA = F.l1_loss(image_fused * gradient_A_Mask, Fu_image_maskA_A)


        Fu_image_maskB_B = image_B * gradient_B_Mask
        Loss_intensity_maskB = F.l1_loss(image_fused * gradient_B_Mask, Fu_image_maskB_B)

        return Loss_intensity_maskA + Loss_intensity_maskB


class fusion_loss_med(nn.Module):
    def __init__(self):
        super(fusion_loss_med, self).__init__()
        self.FusionLoss = Fusionloss()
        
    def forward(self, image_fused, image_A, image_B, opt):
        image_fused = image_fused["pred"]
        gradient_loss, intense_loss, perceptual_loss, color_loss = self.FusionLoss(image_A, image_B, image_fused)
        loss_sbatch = gradient_loss * opt["Loss"]["lr_gradient_loss"] + \
                        intense_loss * opt["Loss"]["lr_intense_loss"] \
                        + color_loss * opt["Loss"]["lr_color_loss"] \
                        + perceptual_loss * opt["Loss"]["lr_perceptual_loss"]
        # print(f"gradient_loss:{gradient_loss}")
        # print(f"intense_loss:{intense_loss}")
        # print(f"perceptual_loss:{perceptual_loss}")
        # print(f"color_loss:{color_loss}")
        return loss_sbatch
