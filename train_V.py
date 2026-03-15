import os
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import random
import time
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from utils import (
    train_ids, test_ids, BATCH_SIZE, Stride_Size, WEIGHTS, WINDOW_SIZE,
    N_CLASSES, DATASET, DATA_FOLDER, DSM_FOLDER, LABEL_FOLDER, ERODED_FOLDER,
    ISPRS_dataset, CACHE, accuracy, metrics, count_sliding_window, 
    grouper, sliding_window, convert_from_color, io
)
from torch.autograd import Variable
from IPython.display import clear_output
from model.vitcross_seg_modeling import VisionTransformer as ViT_seg
from model.vitcross_seg_modeling import CONFIGS as CONFIGS_ViT_seg
try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from pynvml import *
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(int(os.environ["CUDA_VISIBLE_DEVICES"]))
print("Device :", nvmlDeviceGetName(handle))

config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
config_vit.n_classes = 6
config_vit.n_skip = 3
config_vit.patches.grid = (int(256 / 16), int(256 / 16))
net = ViT_seg(config_vit, img_size=256, num_classes=6).cuda()
net.load_from(weights=np.load(config_vit.pretrained_path))
params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print(params)
# Load the datasets

print("training : ", train_ids)
print("testing : ", test_ids)
print("BATCH_SIZE: ", BATCH_SIZE)
print("Stride Size: ", Stride_Size)
IGNORE_INDEX = 6
LOW_VEG_INDEX = 2
LOW_VEG_BOOST = 2.0
CE_LOSS_WEIGHT = 1.0
DICE_LOSS_WEIGHT = 1.0

train_set = ISPRS_dataset(train_ids, cache=CACHE, label_files=ERODED_FOLDER)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)

def compute_class_weights(label_files, num_classes, ignore_index=None):
    counts = np.zeros(num_classes, dtype=np.int64)
    for path in label_files:
        label = np.asarray(convert_from_color(io.imread(path)), dtype='int64')
        if ignore_index is not None:
            label = label[label != ignore_index]
        for c in range(num_classes):
            counts[c] += np.count_nonzero(label == c)
    counts = np.maximum(counts, 1)
    freqs = counts / np.sum(counts)
    weights = 1.0 / np.sqrt(freqs)
    weights = weights / np.mean(weights)
    return torch.from_numpy(weights.astype(np.float32))

TRAIN_CLASS_WEIGHTS = compute_class_weights(train_set.label_files, N_CLASSES, ignore_index=IGNORE_INDEX)
TRAIN_CLASS_WEIGHTS[LOW_VEG_INDEX] = TRAIN_CLASS_WEIGHTS[LOW_VEG_INDEX] * LOW_VEG_BOOST

# 优化器配置改进
base_lr = 2e-4  # 降低学习率以获得更稳定的收敛
params_dict = dict(net.named_parameters())

# 更细粒度的参数分组
encoder_params = []
decoder_params = []
offset_generator_params = []

for key, value in params_dict.items():
    if '_D' in key:
        # Decoder weights - 使用标准学习率
        decoder_params.append(value)
    elif 'offset_generator' in key:
        # Offset Generator参数 - 使用稍高的学习率以加速特征优化
        offset_generator_params.append(value)
    else:
        # Encoder weights - 使用较低的学习率
        encoder_params.append(value)

# 创建参数组
params = [
    {'params': encoder_params, 'lr': base_lr / 3, 'weight_decay': 0.0005},
    {'params': decoder_params, 'lr': base_lr, 'weight_decay': 0.0003},
    {'params': offset_generator_params, 'lr': base_lr * 1.5, 'weight_decay': 0.0001}
]

# 使用改进的优化器
optimizer = optim.AdamW(
    params, 
    lr=base_lr, 
    betas=(0.9, 0.999),  # 更稳定的动量参数
    eps=1e-8,  # 更小的epsilon避免数值问题
    weight_decay=0.0005
)

# 改进的学习率调度器 - 防止过拟合
# 使用更温和的Cosine Annealing LR
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=50,  # 更长的周期，避免频繁重启
    eta_min=1e-6  # 最小学习率
)

# 或者使用更精细的MultiStep调度器（备选方案）
# scheduler = optim.lr_scheduler.MultiStepLR(
#     optimizer, 
#     milestones=[20, 40, 60, 80],  # 更密集的里程碑
#     gamma=0.7  # 更温和的衰减
# )


class DiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=None, weight=None, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps
        if weight is not None:
            self.register_buffer('weight', weight)
        else:
            self.weight = None

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        target = target.long()
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            target = target.clone()
            target[~mask] = 0
            mask = mask.unsqueeze(1).float()
        else:
            mask = None

        one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        if mask is not None:
            probs = probs * mask
            one_hot = one_hot * mask

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        cardinality = probs.sum(dims) + one_hot.sum(dims)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        loss = 1.0 - dice

        if self.weight is not None:
            w = self.weight
            loss = loss * w / (w.sum() + self.eps)
            return loss.sum()
        return loss.mean()


def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    # Use the network on the test set
    ## Potsdam
    if DATASET == 'Potsdam':
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, :3], dtype='float32') for id in test_ids)
        # test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, (3, 0, 1, 2)][:, :, :3], dtype='float32') for id in test_ids)
    ## Vaihingen
    else:
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_dsms = (np.asarray(io.imread(DSM_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    all_preds = []
    all_gts = []

    # Switch the network to inference mode
    with torch.no_grad():
        for img, dsm, gt, gt_e in tqdm(zip(test_images, test_dsms, test_labels, eroded_labels), total=len(test_ids), leave=False):
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))

            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(
                    tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                        leave=False)):
                # Build the tensor
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = torch.from_numpy(image_patches).cuda()

                min = np.min(dsm)
                max = np.max(dsm)
                dsm = (dsm - min) / (max - min)
                dsm_patches = [np.copy(dsm[x:x + w, y:y + h]) for x, y, w, h in coords]
                dsm_patches = np.asarray(dsm_patches)
                dsm_patches = torch.from_numpy(dsm_patches).cuda()

                # Do the inference
                outs = net(image_patches, dsm_patches)
                outs = outs.data.cpu().numpy()

                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del (outs)

            pred = np.argmax(pred, axis=-1)
            all_preds.append(pred)
            all_gts.append(gt_e)
            clear_output()
            
    accuracy, miou = metrics(np.concatenate([p.ravel() for p in all_preds]),
                             np.concatenate([p.ravel() for p in all_gts]).ravel())
    if all:
        return miou, all_preds, all_gts
    else:
        return miou


def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=5, patience=10):
    # 确保保存目录存在
    if not os.path.exists('./resultsv_se_ablation/'):
        os.makedirs('./resultsv_se_ablation/')
        
    # 改进的损失记录
    max_iterations = epochs * len(train_loader)
    losses = np.zeros(max_iterations)
    mean_losses = np.zeros(max_iterations)
    weights = weights.cuda()

    # 使用改进的损失函数组合 - 添加CutMix正则化
    criterion_ce = nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE_INDEX)
    criterion_dice = DiceLoss(num_classes=N_CLASSES, ignore_index=IGNORE_INDEX, weight=weights)
    iter_ = 0
    miou_best = 0.0  # 改为MIoU作为最佳指标
    patience_counter = 0  # 早停计数器
    
    # 记录训练开始时间
    train_start_time = time.time()
    
    # ClassMix参数
    classmix_prob = 0.3  # ClassMix应用概率
    classmix_alpha = 1.0  # Beta分布参数
    
    # 梯度累积参数
    accumulation_steps = 2  # 每2个batch更新一次参数

    for e in range(1, epochs + 1):
        net.train()
        epoch_loss = 0.0
        
        for batch_idx, (data, dsm, target) in enumerate(train_loader):
            data, dsm, target = data.cuda(), dsm.cuda(), target.cuda()
            
            # ClassMix数据增强
            if random.random() < classmix_prob:  # 应用ClassMix的概率
                batch_size = data.size(0)
                index = torch.randperm(batch_size).cuda()
                
                # 获取图像尺寸
                h, w = data.size(2), data.size(3)
                
                # 生成随机混合掩码
                lam = np.random.beta(classmix_alpha, classmix_alpha)
                mask_size = int(h * w * lam)
                
                # 创建随机位置掩码
                mask = torch.zeros(h, w, device='cuda')
                mask_indices = torch.randperm(h * w)[:mask_size]
                mask.view(-1)[mask_indices] = 1
                
                # 扩展掩码到batch和通道维度
                mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 3, h, w)
                
                # 应用ClassMix：基于掩码混合图像和标签
                mixed_data = data.clone()
                mixed_data[mask.bool()] = data[index][mask.bool()]
                
                # 创建混合标签（标签掩码需要调整形状）
                target_mask = mask[:, 0:1, :, :].expand(batch_size, 1, h, w)
                mixed_target = target.clone()
                mixed_target[target_mask.squeeze(1).bool()] = target[index][target_mask.squeeze(1).bool()]
                
                # 前向传播
                output = net(mixed_data, dsm)
                loss = CE_LOSS_WEIGHT * criterion_ce(output, mixed_target) + DICE_LOSS_WEIGHT * criterion_dice(output, mixed_target)
            else:
                # 标准前向传播
                output = net(data, dsm)
                loss = CE_LOSS_WEIGHT * criterion_ce(output, target) + DICE_LOSS_WEIGHT * criterion_dice(output, target)
            
            # 梯度累积
            loss = loss / accumulation_steps
            loss.backward()
            
            # 每accumulation_steps个batch更新一次参数
            if (batch_idx + 1) % accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * accumulation_steps
            losses[iter_] = loss.item() * accumulation_steps
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])
            
            iter_ += 1
            
            if iter_ % 100 == 0:
                clear_output()
                rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                current_lr = optimizer.param_groups[0]['lr']
                print('Train (epoch {}/{}) [{}/{}] LR: {:.2e} Loss: {:.6f} Acc: {:.2f}%'.format(
                    e, epochs, batch_idx, len(train_loader), current_lr, 
                    loss.item() * accumulation_steps, accuracy(pred, gt)))
            
            del (data, target, loss)
        
        # 在每个epoch结束时更新学习率
        if scheduler is not None:
            scheduler.step()
            
        # 打印epoch统计信息
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f'Epoch {e}/{epochs} - Average Loss: {avg_epoch_loss:.4f}')

        # 每5个epoch验证并保存一次模型
        if e % save_epoch == 0:
            net.eval()
            miou = test(net, test_ids, all=False, stride=Stride_Size)
            net.train()
            
            # 计算已用时间
            elapsed_time = time.time() - train_start_time
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            seconds = int(elapsed_time % 60)
            
            # 保存当前epoch的模型
            torch.save(net.state_dict(), './resultsv_se_ablation/segnet256_epoch{}_{:.4f}.pth'.format(e, miou))
            print(f'模型已保存: segnet256_epoch{e}_{miou:.4f}.pth')
            print(f'已训练时间: {hours}小时 {minutes}分钟 {seconds}秒')
            
            # 早停机制
            if miou > miou_best:
                torch.save(net.state_dict(), './resultsv_se_ablation/segnet256_best.pth')
                miou_best = miou
                patience_counter = 0  # 重置计数器
                print(f'新的最佳模型已保存: segnet256_best.pth (MIoU: {miou:.4f})')
            else:
                patience_counter += 1
                print(f'验证MIoU未提升，早停计数器: {patience_counter}/{patience}')
                
                # 如果连续patience个epoch没有提升，则提前停止
                if patience_counter >= patience:
                    total_time = time.time() - train_start_time
                    hours = int(total_time // 3600)
                    minutes = int((total_time % 3600) // 60)
                    seconds = int(total_time % 60)
                    print(f'早停触发！在第{e}个epoch停止训练')
                    print(f'最佳MIoU: {miou_best:.4f}')
                    print(f'总训练时间: {hours}小时 {minutes}分钟 {seconds}秒')
                    return
    
    # 训练完成，计算总时间
    total_time = time.time() - train_start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    print(f'最佳MIoU: {miou_best:.4f}')
    print(f'总训练时间: {hours}小时 {minutes}分钟 {seconds}秒')

#####   train   ####
train(net, optimizer, 100, scheduler, weights=TRAIN_CLASS_WEIGHTS)

#####   test   ####
# net.load_state_dict(torch.load('YOUR_MODEL'))
# net.eval()
# miou, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
# print("MIoU: ", miou)
# for p, id_ in zip(all_preds, test_ids):
#     img = convert_to_color(p)
#     # plt.imshow(img) and plt.show()
#     io.imsave('./results/inference_tile{}.png'.format(id_), img)