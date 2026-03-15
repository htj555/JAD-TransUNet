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
    grouper, sliding_window, convert_from_color, io,
    MMRSLikeAug, MMRSLikeVal, PatchDatasetWithMosaic, compute_dsm_gradients,
    LABELS  # ⭐ 添加LABELS导入
)
from torch.autograd import Variable
from IPython.display import clear_output
from model.vitcross_seg_modeling import VisionTransformer as ViT_seg
from model.vitcross_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from torch.cuda.amp import autocast, GradScaler
try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener
os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # ⭐ 切换到5号卡
from pynvml import *

# ⭐ 混合精度训练配置
USE_AMP = True  # 启用混合精度训练以节省显存

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(int(os.environ["CUDA_VISIBLE_DEVICES"]))
print("Device :", nvmlDeviceGetName(handle))

config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
config_vit.n_classes = 6
config_vit.n_skip = 3
img_size = WINDOW_SIZE[0]  # 使用实际的窗口大小
config_vit.patches.grid = (img_size // 16, img_size // 16)
net = ViT_seg(config_vit, img_size=img_size, num_classes=6).cuda()
net.load_from(weights=np.load(config_vit.pretrained_path))
params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print(params)
# Load the datasets - 使用切割数据集

# ⭐ 注意：train_ids和test_ids仅用于在原始大图上测试
# 训练和验证使用切割数据集，会自动读取文件夹下的所有文件
print("Dataset: ", DATASET)
print("BATCH_SIZE: ", BATCH_SIZE)
print("Stride Size: ", Stride_Size)
print("Note: Training on cropped patches, not using train_ids/test_ids")

# ⭐ 定义切割数据集路径（与train_twoStage.py一致）
PATCH_ROOT = "/home/htj107552403789/寒假突击/6、FTransUNet-1.1 - 副本+数据增强 - 切割数据集 - p +浅层特征融合改进/dataset/Potsdam_patches"
print("Loading cropped dataset from:", PATCH_ROOT)

# ⭐ 创建数据增强器
train_transform = MMRSLikeAug(
    crop_size=WINDOW_SIZE[0],
    scale_list=(0.75, 1.0, 1.25, 1.5),
    max_ratio=0.75,
    ignore_index=255,
    hflip_p=0.5,
    vflip_p=0.5,
    rotate90_p=0.5,
    use_imagenet_norm=True,
)

# ⭐ 使用切割数据集
train_set = PatchDatasetWithMosaic(
    root=os.path.join(PATCH_ROOT, "train"),
    transform=train_transform,
    mosaic_ratio=0.25,
    is_train=True,
    num_classes=N_CLASSES,
    ignore_index=255,
    epoch_len=4000,
    mosaic_seam_ignore_width=3
)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=4,  # ⭐ 恢复到4（5号卡有24GB显存）
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)

# 验证集
val_transform = MMRSLikeVal(
    crop_size=WINDOW_SIZE[0],
    ignore_index=255,
    use_imagenet_norm=True
)
val_set = PatchDatasetWithMosaic(
    root=os.path.join(PATCH_ROOT, "test"),
    transform=val_transform,
    mosaic_ratio=0.0,
    is_train=False,
    num_classes=N_CLASSES,
    ignore_index=255
)

# 优化器配置改进
base_lr = 2e-4  # 降低学习率以获得更稳定的收敛
params_dict = dict(net.named_parameters())

# 更细粒度的参数分组
encoder_params = []
decoder_params = []
local_consistency_optimizer_params = []

for key, value in params_dict.items():
    if '_D' in key:
        # Decoder weights - 使用标准学习率
        decoder_params.append(value)
    elif 'local_consistency_optimizer' in key:
        # 局部一致性优化器参数 - 使用稍高的学习率以加速特征优化
        local_consistency_optimizer_params.append(value)
    else:
        # Encoder weights - 使用较低的学习率
        encoder_params.append(value)

# 创建参数组
params = [
    {'params': encoder_params, 'lr': base_lr / 3, 'weight_decay': 0.0005},
    {'params': decoder_params, 'lr': base_lr, 'weight_decay': 0.0003},
    {'params': local_consistency_optimizer_params, 'lr': base_lr * 1.5, 'weight_decay': 0.0001}
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


def eval_on_patches(net, dataset, num_classes, batch_size=4, num_workers=4):
    """在切割的patch数据集上评估模型"""
    from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers, pin_memory=True)
    
    # 使用MetricsCalculator进行分块计算
    class MetricsCalculator:
        def __init__(self, num_classes, chunk_size=5000000):
            self.num_classes = num_classes
            self.chunk_size = chunk_size
            self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

        def update(self, predictions, targets):
            predictions = predictions.flatten()
            targets = targets.flatten()

            total_pixels = len(predictions)
            for start_idx in range(0, total_pixels, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, total_pixels)

                pred_chunk = predictions[start_idx:end_idx]
                target_chunk = targets[start_idx:end_idx]

                valid_mask = (target_chunk >= 0) & (target_chunk < self.num_classes)
                pred_chunk = pred_chunk[valid_mask]
                target_chunk = target_chunk[valid_mask]

                chunk_cm = sklearn_confusion_matrix(
                    target_chunk, pred_chunk,
                    labels=list(range(self.num_classes))
                )

                self.confusion_matrix += chunk_cm

        def compute_metrics(self):
            cm = self.confusion_matrix
            
            # ⭐ 打印详细的混淆矩阵和指标
            print("\nConfusion matrix :")
            print(cm)
            
            # 计算总体准确率
            total = np.sum(cm)
            accuracy = np.sum(np.diag(cm)) / total
            print(f"{total} pixels processed")
            print(f"Total accuracy : {accuracy * 100:.2f}")
            
            # 计算每类准确率
            recall_per_class = np.divide(np.diag(cm), cm.sum(axis=1),
                                         out=np.zeros(self.num_classes, dtype=float),
                                         where=cm.sum(axis=1) != 0)
            for l_id, score in enumerate(recall_per_class):
                print(f"{LABELS[l_id]}: {score:.4f}")
            print("---")
            
            # 计算F1分数
            precision_per_class = np.divide(np.diag(cm), cm.sum(axis=0),
                                            out=np.zeros(self.num_classes, dtype=float),
                                            where=cm.sum(axis=0) != 0)
            f1_scores = np.zeros(self.num_classes)
            for i in range(self.num_classes):
                if recall_per_class[i] + precision_per_class[i] > 0:
                    f1_scores[i] = 2 * (precision_per_class[i] * recall_per_class[i]) / (precision_per_class[i] + recall_per_class[i])
            
            print("F1Score :")
            for l_id, score in enumerate(f1_scores):
                print(f"{LABELS[l_id]}: {score:.4f}")
            print(f'mean F1Score: {np.mean(f1_scores[:5]):.4f}')
            print("---")
            
            # 计算Kappa系数
            pa = np.trace(cm) / total
            pe = np.sum(cm.sum(axis=0) * cm.sum(axis=1)) / (total * total)
            kappa = (pa - pe) / (1 - pe) if pe != 1 else 0
            print(f"Kappa: {kappa:.4f}")
            
            # 计算IoU
            intersection = np.diag(cm)
            union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
            iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float),
                            where=union != 0)
            
            print(iou)
            mean_iou = np.mean(iou[:5])  # 只计算前5类
            mean_iou_all = np.mean(iou)
            print(f'mean MIoU: {mean_iou:.4f}')
            print("---")

            return {
                'overall_acc': accuracy * 100,
                'accuracy': accuracy * 100,
                'mean_iou': mean_iou * 100,
                'mean_iou_all': mean_iou_all * 100,
                'per_class_iou': iou * 100,
                'per_class_acc': recall_per_class * 100,
                'precision': np.mean(precision_per_class) * 100,
                'recall': np.mean(recall_per_class) * 100,
                'confusion_matrix': cm
            }

        def reset(self):
            self.confusion_matrix.fill(0)
    
    meter = MetricsCalculator(num_classes=num_classes, chunk_size=5000000)

    net.eval()
    with torch.no_grad():
        for img, dsm, gt in loader:
            img = img.cuda(non_blocking=True)
            dsm = dsm.cuda(non_blocking=True)
            gt = gt.squeeze(1).cuda(non_blocking=True)  # (B,H,W)

            logits = net(img, dsm)

            pred = logits.argmax(dim=1).cpu().numpy()
            gt_np = gt.cpu().numpy()
            meter.update(pred, gt_np)

    return meter.compute_metrics()


def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    """测试函数 - 支持3通道DSM"""
    if DATASET == 'Potsdam':
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, :3], dtype='float32')
                       for id in test_ids)
    else:
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32')
                       for id in test_ids)

    # ⭐ 加载原始DSM（不预处理，在推理时计算梯度）
    test_dsms = (np.asarray(io.imread(DSM_FOLDER.format(id)), dtype='float32')
                 for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8')
                   for id in test_ids)
    eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id)))
                     for id in test_ids)

    all_preds = []
    all_gts = []

    net.eval()
    with torch.no_grad():
        for img, dsm_raw, gt, gt_e in tqdm(zip(test_images, test_dsms, test_labels, eroded_labels),
                                           total=len(test_ids),
                                           desc="Testing images"):
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))

            # ⭐⭐⭐ 计算整图的3通道DSM特征 ⭐⭐⭐
            dsm_3ch = compute_dsm_gradients(dsm_raw)  # (3, H, W)

            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size

            for i, coords in enumerate(
                    tqdm(
                        grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)),
                        total=total,
                        desc="Inference",
                        leave=False
                    )
            ):
                # RGB patches
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = torch.from_numpy(image_patches).float().cuda()
                mean = torch.tensor([0.485, 0.456, 0.406], device=image_patches.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=image_patches.device).view(1, 3, 1, 1)
                image_patches = (image_patches - mean) / std

                # ⭐ DSM patches - 3通道
                dsm_patches = [np.copy(dsm_3ch[:, x:x + w, y:y + h]) for x, y, w, h in coords]
                dsm_patches = np.asarray(dsm_patches)  # (B, 3, H, W)
                dsm_patches = torch.from_numpy(dsm_patches).cuda()

                outs = net(image_patches, dsm_patches)
                outs = F.softmax(outs, dim=1)
                outs = outs.data.cpu().numpy()

                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out

                del outs

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

    # 使用改进的损失函数组合 - 添加ignore_index
    criterion_ce = nn.CrossEntropyLoss(weight=weights, ignore_index=255)
    iter_ = 0
    miou_best = 0.0  # 改为MIoU作为最佳指标
    patience_counter = 0  # 早停计数器
    
    # 记录训练开始时间
    train_start_time = time.time()
    
    # ⭐ 初始化梯度缩放器（用于混合精度训练）
    scaler = GradScaler(enabled=USE_AMP)
    
    # ClassMix参数
    classmix_prob = 0.3  # ClassMix应用概率
    classmix_alpha = 1.0  # Beta分布参数
    
    # 梯度累积参数（与train_twoStage.py一致）
    accumulation_steps = 4  # ⭐ batch_size=4, 有效batch_size=16
    effective_batch_size = 4 * accumulation_steps  # 有效batch size = 16
    
    print(f"\n{'='*60}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"GPU: CUDA:5 (24GB)")
    print(f"Physical Batch Size: 4")
    print(f"Accumulation Steps: {accumulation_steps}")
    print(f"Effective Batch Size: {effective_batch_size}")
    print(f"Mixed Precision (AMP): {'Enabled' if USE_AMP else 'Disabled'}")
    print(f"ClassMix Probability: {classmix_prob}")
    print(f"{'='*60}\n")

    for e in range(1, epochs + 1):
        net.train()
        epoch_loss = 0.0
        
        for batch_idx, (data, dsm, target) in enumerate(train_loader):
            data, dsm, target = data.cuda(), dsm.cuda(), target.cuda()
            
            # ⭐ 调试：打印原始形状
            if batch_idx == 0:
                print(f"Original shapes - data: {data.shape}, dsm: {dsm.shape}, target: {target.shape}")
            
            # ⭐ 修复维度：target从[B,1,H,W]变为[B,H,W]
            if target.dim() == 4 and target.size(1) == 1:
                target = target.squeeze(1)
            
            # ⭐ 修复维度：dsm从[B,1,3,H,W]变为[B,3,H,W]
            if dsm.dim() == 5 and dsm.size(1) == 1:
                dsm = dsm.squeeze(1)
            elif dsm.dim() == 4 and dsm.size(1) == 1:
                # 如果是[B,1,H,W]，需要先squeeze再计算梯度
                print(f"Warning: DSM has unexpected shape {dsm.shape}, fixing...")
                dsm = dsm.squeeze(1)
            
            if batch_idx == 0:
                print(f"After fix - data: {data.shape}, dsm: {dsm.shape}, target: {target.shape}")
            
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
                
                # ⭐ 调试：打印ClassMix后的形状
                if batch_idx == 0:
                    print(f"ClassMix - mixed_data: {mixed_data.shape}, dsm: {dsm.shape}, mixed_target: {mixed_target.shape}")
                
                # ⭐ 使用混合精度训练
                with autocast(enabled=USE_AMP):
                    output = net(mixed_data, dsm)
                    loss = criterion_ce(output, mixed_target)
            else:
                # 标准前向传播
                if batch_idx == 0:
                    print(f"No ClassMix - data: {data.shape}, dsm: {dsm.shape}, target: {target.shape}")
                
                # ⭐ 使用混合精度训练
                with autocast(enabled=USE_AMP):
                    output = net(data, dsm)
                    loss = criterion_ce(output, target)
            
            # 梯度累积
            loss = loss / accumulation_steps
            
            # ⭐ 使用scaler进行反向传播
            scaler.scale(loss).backward()
            
            # 每accumulation_steps个batch更新一次参数
            if (batch_idx + 1) % accumulation_steps == 0:
                # ⭐ 梯度裁剪（在scaler.unscale_之后）
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                
                # ⭐ 使用scaler更新参数
                scaler.step(optimizer)
                scaler.update()
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
            
            # ⭐ 创建日志文件记录详细测试结果
            log_file = f'./resultsv_se_ablation/test_results_epoch{e}.txt'
            import sys
            
            # 保存原始stdout
            original_stdout = sys.stdout
            
            # 同时输出到控制台和文件
            class Logger:
                def __init__(self, filename):
                    self.terminal = original_stdout
                    self.log = open(filename, 'w', encoding='utf-8')
                
                def write(self, message):
                    self.terminal.write(message)
                    self.log.write(message)
                    self.log.flush()
                
                def flush(self):
                    self.terminal.flush()
                    self.log.flush()
            
            sys.stdout = Logger(log_file)
            
            print("=" * 70)
            print(f"Epoch {e}/{epochs} - 验证结果")
            print("=" * 70)
            
            # ⭐ 修改：使用eval_on_patches在切割数据集上评估
            val_metrics = eval_on_patches(net, val_set, num_classes=N_CLASSES, batch_size=BATCH_SIZE, num_workers=4)
            miou = val_metrics['mean_iou']
            val_acc = val_metrics['accuracy']
            
            # 恢复stdout
            sys.stdout = original_stdout
            
            net.train()
            
            # 计算已用时间
            elapsed_time = time.time() - train_start_time
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            seconds = int(elapsed_time % 60)
            
            # 保存当前epoch的模型
            torch.save(net.state_dict(), './resultsv_se_ablation/segnet256_epoch{}_{:.4f}.pth'.format(e, miou))
            print(f'模型已保存: segnet256_epoch{e}_{miou:.4f}.pth')
            print(f'验证准确率: {val_acc:.2f}%, 验证mIoU: {miou:.2f}%')
            print(f'详细结果已保存到: {log_file}')
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
train(net, optimizer, 100, scheduler)

#####   test   ####
# ⭐ 在切割数据集上测试
# net.load_state_dict(torch.load('./resultsv_se_ablation/segnet256_best.pth'))
# net.eval()
# test_metrics = eval_on_patches(net, val_set, num_classes=N_CLASSES, batch_size=BATCH_SIZE, num_workers=4)
# print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
# print(f"Test mIoU: {test_metrics['mean_iou']:.2f}%")
# print(f"Per-class IoU: {test_metrics['per_class_iou']}")

# ⭐ 或者在原始大图上测试（使用test函数）
# net.load_state_dict(torch.load('./resultsv_se_ablation/segnet256_best.pth'))
# net.eval()
# miou, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
# print("MIoU: ", miou)
# for p, id_ in zip(all_preds, test_ids):
#     img = convert_to_color(p)
#     io.imsave('./results/inference_tile{}.png'.format(id_), img)