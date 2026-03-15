import os
import numpy as np
import torch
from tqdm import tqdm
from skimage import io

# 导入你的 utils 中的变量和函数
from utils import (
    test_ids, BATCH_SIZE, Stride_Size, WINDOW_SIZE, N_CLASSES,
    DATASET, DATA_FOLDER, DSM_FOLDER, LABEL_FOLDER,
    count_sliding_window, grouper, sliding_window, convert_to_color
)

# 导入你的模型
from model.vitcross_seg_modeling import VisionTransformer as ViT_seg
from model.vitcross_seg_modeling import CONFIGS as CONFIGS_ViT_seg

def main():
    # ================= 配置路径 =================
    # 根据你提供的路径，设置最佳模型权重的绝对路径
    model_weight_path = "/home/htj107552403789/寒假突击/4、FTransUNet-1.1 - 副本+数据增强+编码器浅层特征融合2/resultsv_se_ablation/segnet256_epoch35_0.8528.pth"
    
    # 预测图和真值图的保存目录
    output_dir = "./results_visualization/"
    os.makedirs(output_dir, exist_ok=True)

    # ================= 初始化硬件 =================
    os.environ["CUDA_VISIBLE_DEVICES"] = "7" # 保持和你训练时一致的显卡
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ================= 初始化模型 =================
    print("Loading model configuration and weights...")
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = N_CLASSES
    config_vit.n_skip = 3
    config_vit.patches.grid = (int(256 / 16), int(256 / 16))
    
    net = ViT_seg(config_vit, img_size=256, num_classes=N_CLASSES)
    
    # 加载训练好的权重
    if not os.path.exists(model_weight_path):
        raise FileNotFoundError(f"找不到模型权重文件，请检查路径: {model_weight_path}")
    
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net.to(device)
    net.eval()
    print("Model loaded successfully!")

    # ================= 加载测试数据 =================
    print(f"Testing IDs: {test_ids}")
    
    if DATASET == 'Potsdam':
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, :3], dtype='float32') for id in test_ids)
    else:
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
        
    test_dsms = (np.asarray(io.imread(DSM_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)

    # ================= 开始推理 =================
    with torch.no_grad():
        for img, dsm, gt, id_ in zip(test_images, test_dsms, test_labels, test_ids):
            print(f"\nProcessing Tile ID: {id_} ...")
            
            # 初始化一个全零数组用于累加预测结果
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))
            
            # 计算需要滑动窗口的总数，用于显示进度条
            total_windows = count_sliding_window(img, step=Stride_Size, window_size=WINDOW_SIZE) // BATCH_SIZE
            
            for coords in tqdm(grouper(BATCH_SIZE, sliding_window(img, step=Stride_Size, window_size=WINDOW_SIZE)), 
                               total=total_windows, leave=False):
                
                # 提取 Image patch
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = torch.from_numpy(image_patches).to(device)

                # 提取 DSM patch 并归一化
                min_val = np.min(dsm)
                max_val = np.max(dsm)
                dsm_norm = (dsm - min_val) / (max_val - min_val) if max_val > min_val else dsm
                dsm_patches = [np.copy(dsm_norm[x:x + w, y:y + h]) for x, y, w, h in coords]
                dsm_patches = np.asarray(dsm_patches)
                dsm_patches = torch.from_numpy(dsm_patches).to(device)

                # 模型前向推理
                outs = net(image_patches, dsm_patches)
                outs = outs.data.cpu().numpy()

                # 将预测概率填回到整图的对应位置
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                    
            # 获取最终类别 (在通道维度上取 Argmax)
            pred_classes = np.argmax(pred, axis=-1)
            
            # ================= 保存可视化结果 =================
            # 1. 转换预测结果为RGB
            pred_rgb = convert_to_color(pred_classes)
            pred_save_path = os.path.join(output_dir, f'pred_tile_{id_}.png')
            io.imsave(pred_save_path, pred_rgb)
            
            # 2. 转换真值(Ground Truth)为RGB (由于你的 GT 加载后是RGB格式，这里直接保存即可，或者按需转换)
            # 注意: 如果你的 gt 是索引形式的单通道图，请取消下面第一行的注释
            # gt_rgb = convert_to_color(gt) 
            gt_save_path = os.path.join(output_dir, f'gt_tile_{id_}.png')
            io.imsave(gt_save_path, gt) # 如果你的 label 本身就是彩色图的话直接保存
            
            print(f"Saved: {pred_save_path} and {gt_save_path}")

    print("\nAll visualizations have been successfully generated!")

if __name__ == '__main__':
    main()