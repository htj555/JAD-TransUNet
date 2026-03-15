import numpy as np
from sklearn.metrics import confusion_matrix
import random
import torch
import torch.nn.functional as F
import itertools
from torchvision.utils import make_grid
from PIL import Image
from skimage import io
import os
import scipy.ndimage as ndimage
import torch.nn as nn
import cv2
import albumentations as A
from glob import glob

# Parameters
## SwinFusion
# WINDOW_SIZE = (64, 64) # Patch size
WINDOW_SIZE = (512, 512) # Patch size

STRIDE = 32 # Stride for testing
IN_CHANNELS = 3 # Number of input channels (e.g. RGB)
FOLDER = "/home/htj107552403789/寒假突击/FTransUNet-1.1 - 副本+数据增强 - 切割数据集/dataset/" # Replace with your "/path/to/the/ISPRS/dataset/folder/"
BATCH_SIZE = 4 # Number of samples in a mini-batch

LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"] # Label names
N_CLASSES = len(LABELS) # Number of classes
WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
CACHE = True # Store the dataset in-memory

# ⭐ 注意：使用切割数据集时，不需要指定train_ids和test_ids
# 数据集加载器会自动读取对应文件夹下的所有文件
# train_ids和test_ids仅用于在原始大图上测试时使用

# Vaihingen数据集配置（用于原始大图测试）
# train_ids = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37']
# test_ids = ['5', '21', '15', '30']
# DATASET = 'Vaihingen'

# Potsdam数据集配置（用于原始大图测试）
train_ids = ['6_10', '7_10', '2_12', '3_11', '2_10', '7_8', '5_10', '3_12', '5_12', '7_11', '7_9', '6_9', '7_7',
             '4_12', '6_8', '6_12', '6_7', '4_11']
test_ids = ['4_10', '5_11', '2_11', '3_10', '6_11', '7_12']
DATASET = 'Potsdam'
Stride_Size = 128
MAIN_FOLDER = FOLDER + 'Potsdam/'
DATA_FOLDER = MAIN_FOLDER + '4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif'
DSM_FOLDER = MAIN_FOLDER + '1_DSM_normalisation/dsm_potsdam_{}_normalized_lastools.jpg'
LABEL_FOLDER = MAIN_FOLDER + '5_Labels_for_participants/top_potsdam_{}_label.tif'
ERODED_FOLDER = MAIN_FOLDER + '5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif'

# ISPRS color palette
# Let's define the standard ISPRS color palette
palette = {0 : (255, 255, 255), # Impervious surfaces (white)
           1 : (0, 0, 255),     # Buildings (blue)
           2 : (0, 255, 255),   # Low vegetation (cyan)
           3 : (0, 255, 0),     # Trees (green)
           4 : (255, 255, 0),   # Cars (yellow)
           5 : (255, 0, 0),     # Clutter (red)
           6 : (0, 0, 0)}       # Undefined (black)

invert_palette = {v: k for k, v in palette.items()}

def convert_to_color(arr_2d, palette=palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

def save_img(tensor, name):
    tensor = tensor.cpu() .permute((1, 0, 2, 3))
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (im.data.numpy() * 255.).astype(np.uint8)
    Image.fromarray(im).save(name + '.jpg')


def compute_dsm_gradients(dsm, k=3, sigma=1.0):
    """
    计算DSM的梯度特征，生成3通道DSM
    1) dsm / 255 归一化
    2) gaussian smoothing
    3) sobel-like kernel计算梯度
    4) concat: [grad_x, grad_y, dsm_norm]
    """
    dsm = dsm.astype(np.float32)
    if dsm.max() > 1.5:
        dsm = dsm / 255.0
    dsm = np.clip(dsm, 0.0, 1.0)

    # gaussian平滑
    dsm_blur = ndimage.gaussian_filter(dsm, sigma=sigma)

    # sobel kernel
    r = np.linspace(-(k // 2), k // 2, k)
    x, y = np.meshgrid(r, r)
    denom = (x ** 2 + y ** 2)
    denom[:, k // 2] = 1.0
    sobel = x / denom

    grad_x = np.abs(ndimage.convolve(dsm_blur, sobel, mode="nearest"))
    grad_y = np.abs(ndimage.convolve(dsm_blur, sobel.T, mode="nearest"))

    return np.stack([grad_x, grad_y, dsm], axis=0).astype(np.float32)


class MMRSLikeAug:
    """
    训练用数据增强（对齐MMRSSeg风格）：
    - RandomScale
    - SmartCrop（cat_max_ratio）
    - RandomFlip（H/V）
    - RandomRotate90
    - Normalize（ImageNet or /255）
    """
    def __init__(
        self,
        crop_size=256,
        scale_list=(0.5, 0.75, 1.0, 1.25, 1.5),
        max_ratio=0.75,
        ignore_index=255,
        hflip_p=0.5,
        vflip_p=0.5,
        rotate90_p=0.5,
        use_imagenet_norm=True,
    ):
        self.crop_size = int(crop_size)
        self.scale_list = list(scale_list)
        self.max_ratio = float(max_ratio)
        self.ignore_index = int(ignore_index)

        tfs = []
        if hflip_p > 0:
            tfs.append(A.HorizontalFlip(p=float(hflip_p)))
        if vflip_p > 0:
            tfs.append(A.VerticalFlip(p=float(vflip_p)))
        if rotate90_p > 0:
            tfs.append(A.RandomRotate90(p=float(rotate90_p)))

        if use_imagenet_norm:
            tfs.append(A.Normalize())
        else:
            tfs.append(A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0))

        self.albu = A.Compose(tfs, additional_targets={"dsm": "mask"})

    def _ensure_u8(self, img):
        if img.dtype == np.uint8:
            return img
        img = img.astype(np.float32)
        if img.max() <= 1.5:
            img *= 255.0
        return np.clip(img, 0, 255).astype(np.uint8)

    def _random_scale(self, img_u8, mask, dsm_hwc):
        s = random.choice(self.scale_list)
        h, w = img_u8.shape[:2]
        nh, nw = max(1, int(h * s)), max(1, int(w * s))
        img2 = cv2.resize(img_u8, (nw, nh), interpolation=cv2.INTER_LINEAR)
        mask2 = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        dsm2 = cv2.resize(dsm_hwc, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return img2, mask2, dsm2

    def _pad_if_needed(self, img, mask, dsm_hwc):
        cs = self.crop_size
        h, w = img.shape[:2]
        pad_h = max(cs - h, 0)
        pad_w = max(cs - w, 0)
        if pad_h == 0 and pad_w == 0:
            return img, mask, dsm_hwc

        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.ignore_index)
        dsm_hwc = cv2.copyMakeBorder(dsm_hwc, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
        return img, mask, dsm_hwc

    def _smart_crop(self, img, mask, dsm_hwc):
        img, mask, dsm_hwc = self._pad_if_needed(img, mask, dsm_hwc)

        last = None
        for _ in range(10):
            h, w = img.shape[:2]
            cs = self.crop_size
            y = 0 if h == cs else random.randint(0, h - cs)
            x = 0 if w == cs else random.randint(0, w - cs)

            img_c = img[y:y + cs, x:x + cs]
            mask_c = mask[y:y + cs, x:x + cs]
            dsm_c = dsm_hwc[y:y + cs, x:x + cs]

            labels, cnt = np.unique(mask_c, return_counts=True)
            cnt = cnt[labels != self.ignore_index]
            last = (img_c, mask_c, dsm_c)
            if len(cnt) > 1 and (cnt.max() / cnt.sum()) < self.max_ratio:
                return last
        return last

    def __call__(self, img, dsm_chw, mask):
        img_u8 = self._ensure_u8(img)
        dsm_hwc = np.transpose(dsm_chw, (1, 2, 0)).astype(np.float32)

        img_u8, mask, dsm_hwc = self._random_scale(img_u8, mask, dsm_hwc)
        img_u8, mask, dsm_hwc = self._smart_crop(img_u8, mask, dsm_hwc)

        out = self.albu(image=img_u8, mask=mask, dsm=dsm_hwc)
        img_out = out["image"].astype(np.float32)
        mask_out = out["mask"].astype(np.int64)

        dsm_aug = np.asarray(out["dsm"], dtype=np.float32)
        if dsm_aug.ndim == 3:
            dsm_aug = dsm_aug[..., 0]
        elif dsm_aug.ndim != 2:
            raise ValueError(f"Unexpected dsm shape: {dsm_aug.shape}")

        dsm_out = compute_dsm_gradients(dsm_aug)

        return img_out, dsm_out, mask_out


class MMRSLikeVal:
    """
    验证/测试用：无随机增强
    - 不random scale
    - 不random crop（只做center-crop/pad）
    - 不flip/rotate
    - 只normalize
    """
    def __init__(self, crop_size=256, ignore_index=255, use_imagenet_norm=True):
        self.crop_size = int(crop_size)
        self.ignore_index = int(ignore_index)

        if use_imagenet_norm:
            self.norm = A.Normalize()
        else:
            self.norm = A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0)

    def _ensure_u8(self, img):
        if img.dtype == np.uint8:
            return img
        img = img.astype(np.float32)
        if img.max() <= 1.5:
            img *= 255.0
        return np.clip(img, 0, 255).astype(np.uint8)

    def _pad_if_needed(self, img, mask, dsm_hwc):
        cs = self.crop_size
        h, w = img.shape[:2]
        pad_h = max(cs - h, 0)
        pad_w = max(cs - w, 0)
        if pad_h == 0 and pad_w == 0:
            return img, mask, dsm_hwc

        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.ignore_index)
        dsm_hwc = cv2.copyMakeBorder(dsm_hwc, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
        return img, mask, dsm_hwc

    def _center_crop(self, img, mask, dsm_hwc):
        img, mask, dsm_hwc = self._pad_if_needed(img, mask, dsm_hwc)
        cs = self.crop_size
        h, w = img.shape[:2]
        y = 0 if h == cs else (h - cs) // 2
        x = 0 if w == cs else (w - cs) // 2
        return (
            img[y:y + cs, x:x + cs],
            mask[y:y + cs, x:x + cs],
            dsm_hwc[y:y + cs, x:x + cs],
        )

    def __call__(self, img, dsm_chw, mask):
        img_u8 = self._ensure_u8(img)
        dsm_hwc = np.transpose(dsm_chw, (1, 2, 0)).astype(np.float32)

        img_u8, mask, dsm_hwc = self._center_crop(img_u8, mask, dsm_hwc)

        img_out = self.norm(image=img_u8)["image"].astype(np.float32)
        mask_out = mask.astype(np.int64)

        dsm_aug = np.asarray(dsm_hwc, dtype=np.float32)
        if dsm_aug.ndim == 3:
            dsm_aug = dsm_aug[..., 0]
        elif dsm_aug.ndim != 2:
            raise ValueError(f"Unexpected dsm shape after center_crop: {dsm_aug.shape}")

        dsm_out = compute_dsm_gradients(dsm_aug)

        return img_out, dsm_out, mask_out


class PatchDatasetWithMosaic(torch.utils.data.Dataset):
    """
    切割数据集加载器，支持Mosaic数据增强
    """
    def __init__(self, root, transform=None, mosaic_ratio=0.25, is_train=True, 
                 num_classes=6, ignore_index=255, epoch_len=None, mosaic_seam_ignore_width=0):
        self.root = root
        self.transform = transform
        self.mosaic_ratio = float(mosaic_ratio)
        self.is_train = bool(is_train)
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.epoch_len = epoch_len
        self.mosaic_seam_ignore_width = int(mosaic_seam_ignore_width)

        self.img_dir = os.path.join(root, "images")
        self.dsm_dir = os.path.join(root, "dsms")
        self.mask_dir = os.path.join(root, "masks")
        self.mask_rgb_dir = os.path.join(root, "masks_rgb")
        self.ir_dir = os.path.join(root, "irs")
        self.has_ir = os.path.isdir(self.ir_dir)

        self.img_paths = sorted(
            glob(os.path.join(self.img_dir, "*.tif")) +
            glob(os.path.join(self.img_dir, "*.tiff")) +
            glob(os.path.join(self.img_dir, "*.png")) +
            glob(os.path.join(self.img_dir, "*.jpg"))
        )
        if len(self.img_paths) == 0:
            raise FileNotFoundError(f"No images found: {self.img_dir}")

    def __len__(self):
        if self.is_train and self.epoch_len is not None:
            return int(self.epoch_len)
        return len(self.img_paths)

    def _find(self, folder, stem):
        for suf in (".tif", ".tiff", ".png", ".jpg"):
            p = os.path.join(folder, stem + suf)
            if os.path.exists(p):
                return p
        return None

    def _load_one(self, idx):
        img_path = self.img_paths[idx]
        stem = os.path.splitext(os.path.basename(img_path))[0]

        img = io.imread(img_path)
        if (img.ndim == 3 and img.shape[2] != 3) or img.ndim == 2:
            if img.ndim == 2:
                img = img[..., None]

            if img.shape[2] == 2 and self.has_ir:
                ir_path = self._find(self.ir_dir, stem)
                if ir_path is not None:
                    ir = io.imread(ir_path)
                    if ir.ndim == 3:
                        ir = ir[..., 0]
                    if ir.dtype != np.uint8:
                        ir = ir.astype(np.float32)
                        ir = (ir - ir.min()) / (ir.max() - ir.min() + 1e-6) * 255.0
                        ir = ir.astype(np.uint8)
                    ir = ir[..., None]
                    img = np.concatenate([img.astype(np.uint8), ir], axis=2)
        if img.ndim == 3 and img.shape[2] > 3:
            img = img[:, :, :3]
        img = img.astype(np.uint8)

        mask_path = self._find(self.mask_dir, stem)
        if mask_path is None and os.path.isdir(self.mask_rgb_dir):
            mask_path = self._find(self.mask_rgb_dir, stem)
        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for {stem}")

        mask = io.imread(mask_path)
        if mask.ndim == 3:
            mask = convert_from_color(mask)
        mask = mask.astype(np.int64)
        mask[(mask < 0) | (mask >= self.num_classes)] = self.ignore_index

        dsm_path = self._find(self.dsm_dir, stem)
        if dsm_path is None:
            raise FileNotFoundError(f"DSM not found for {stem}")
        dsm_raw = io.imread(dsm_path).astype(np.float32)
        if dsm_raw.ndim == 3:
            dsm_raw = dsm_raw[..., 0]
        dsm_1ch = dsm_raw[None, ...].astype(np.float32)
        return img, dsm_1ch, mask

    def _mosaic(self, index):
        idxs = [index] + [random.randint(0, len(self.img_paths) - 1) for _ in range(3)]
        a = self._load_one(idxs[0])
        b = self._load_one(idxs[1])
        c = self._load_one(idxs[2])
        d = self._load_one(idxs[3])

        img_a, dsm_a, mask_a = a
        img_b, dsm_b, mask_b = b
        img_c, dsm_c, mask_c = c
        img_d, dsm_d, mask_d = d

        H, W = img_a.shape[:2]

        def _fit(img, dsm, mask):
            if img.shape[0] != H or img.shape[1] != W:
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int64)
                dsm_hwc = np.transpose(dsm, (1,2,0))
                dsm_hwc = cv2.resize(dsm_hwc, (W, H), interpolation=cv2.INTER_LINEAR)
                dsm = np.transpose(dsm_hwc, (2,0,1))
            return img, dsm, mask

        img_b, dsm_b, mask_b = _fit(img_b, dsm_b, mask_b)
        img_c, dsm_c, mask_c = _fit(img_c, dsm_c, mask_c)
        img_d, dsm_d, mask_d = _fit(img_d, dsm_d, mask_d)

        def _rand_crop(img, dsm, mask, hh, ww):
            aug = A.Compose([A.RandomCrop(height=hh, width=ww)], additional_targets={"dsm":"image"})
            out = aug(image=img, mask=mask.astype(np.uint8), dsm=np.transpose(dsm,(1,2,0)))
            img2 = out["image"]
            mask2 = out["mask"].astype(np.int64)
            dsm2 = np.transpose(out["dsm"], (2,0,1)).astype(np.float32)
            return img2, dsm2, mask2

        split_x = random.randint(W//4, 3*W//4)
        split_y = random.randint(H//4, 3*H//4)

        img_a, dsm_a, mask_a = _rand_crop(img_a, dsm_a, mask_a, split_y, split_x)
        img_b, dsm_b, mask_b = _rand_crop(img_b, dsm_b, mask_b, split_y, W-split_x)
        img_c, dsm_c, mask_c = _rand_crop(img_c, dsm_c, mask_c, H-split_y, split_x)
        img_d, dsm_d, mask_d = _rand_crop(img_d, dsm_d, mask_d, H-split_y, W-split_x)

        img = np.concatenate([np.concatenate([img_a, img_b], axis=1),
                              np.concatenate([img_c, img_d], axis=1)], axis=0)

        mask = np.concatenate([np.concatenate([mask_a, mask_b], axis=1),
                               np.concatenate([mask_c, mask_d], axis=1)], axis=0)

        dsm = np.concatenate([np.concatenate([np.transpose(dsm_a,(1,2,0)), np.transpose(dsm_b,(1,2,0))], axis=1),
                              np.concatenate([np.transpose(dsm_c,(1,2,0)), np.transpose(dsm_d,(1,2,0))], axis=1)], axis=0)
        dsm = np.transpose(dsm, (2,0,1)).astype(np.float32)

        mask[(mask < 0) | (mask >= self.num_classes)] = self.ignore_index

        if self.mosaic_seam_ignore_width > 0:
            w = self.mosaic_seam_ignore_width
            h_total, w_total = mask.shape

            x_start = max(0, split_x - w)
            x_end = min(w_total, split_x + w)
            mask[:, x_start:x_end] = self.ignore_index

            y_start = max(0, split_y - w)
            y_end = min(h_total, split_y + w)
            mask[y_start:y_end, :] = self.ignore_index

        return img, dsm, mask

    def __getitem__(self, index):
        if self.is_train and self.epoch_len is not None:
            index = random.randint(0, len(self.img_paths) - 1)

        if self.is_train and random.random() < self.mosaic_ratio:
            img, dsm, mask = self._mosaic(index)
        else:
            img, dsm, mask = self._load_one(index)

        if self.transform is not None:
            img, dsm, mask = self.transform(img, dsm, mask)

        # ⭐ 确保DSM是(3, H, W)格式
        if dsm.ndim == 2:
            # 如果是(H, W)，先计算梯度
            dsm = compute_dsm_gradients(dsm)
        elif dsm.ndim == 3 and dsm.shape[0] == 1:
            # 如果是(1, H, W)，squeeze后计算梯度
            dsm = compute_dsm_gradients(dsm[0])
        elif dsm.ndim == 3 and dsm.shape[0] == 3:
            # 已经是(3, H, W)，直接使用
            pass
        else:
            raise ValueError(f"Unexpected DSM shape: {dsm.shape}")
        
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float()
        dsm_t = torch.from_numpy(dsm).float()
        gt_t = torch.from_numpy(mask[None, ...]).long()
        
        return img_t, dsm_t, gt_t

class ISPRS_dataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_files=DATA_FOLDER, label_files=LABEL_FOLDER,
                 cache=False, augmentation=True):
        super(ISPRS_dataset, self).__init__()

        self.augmentation = augmentation
        self.cache = cache

        # List of files
        self.data_files = [DATA_FOLDER.format(id) for id in ids]
        # self.boundary_files = [BOUNDARY_FOLDER.format(id) for id in ids]
        self.dsm_files = [DSM_FOLDER.format(id) for id in ids]
        self.label_files = [LABEL_FOLDER.format(id) for id in ids]

        # Sanity check : raise an error if some files do not exist
        for f in self.data_files + self.dsm_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))

        # Initialize cache dicts
        self.data_cache_ = {}
        # self.boundary_cache_ = {}
        self.dsm_cache_ = {}
        self.label_cache_ = {}

    def __len__(self):
        # Default epoch size is 10 000 samples
        if DATASET == 'Potsdam':
            return BATCH_SIZE * 1000
        elif DATASET == 'Vaihingen':
            return BATCH_SIZE * 1000
        else:
            return None

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))

        return tuple(results)

    def __getitem__(self, i):
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)

        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Data is normalized in [0, 1]
            ## Potsdam IRRG
            if DATASET == 'Potsdam':
                ## RGB
                data = io.imread(self.data_files[random_idx])[:, :, :3].transpose((2, 0, 1))
                ## IRRG
                # data = io.imread(self.data_files[random_idx])[:, :, (3, 0, 1, 2)][:, :, :3].transpose((2, 0, 1))
                data = 1 / 255 * np.asarray(data, dtype='float32')
            else:
            ## Vaihingen IRRG
                data = io.imread(self.data_files[random_idx])
                data = 1 / 255 * np.asarray(data.transpose((2, 0, 1)), dtype='float32')
            if self.cache:
                self.data_cache_[random_idx] = data
        
        # if random_idx in self.boundary_cache_.keys():
        #     boundary = self.boundary_cache_[random_idx]
        # else:
        #     boundary = np.asarray(io.imread(self.boundary_files[random_idx])) / 255
        #     boundary = boundary.astype(np.int64)
        #     if self.cache:
        #         self.boundary_cache_[random_idx] = boundary

        if random_idx in self.dsm_cache_.keys():
            dsm = self.dsm_cache_[random_idx]
        else:
            # DSM is normalized in [0, 1]
            dsm = np.asarray(io.imread(self.dsm_files[random_idx]), dtype='float32')
            min = np.min(dsm)
            max = np.max(dsm)
            dsm = (dsm - min) / (max - min)
            if self.cache:
                self.dsm_cache_[random_idx] = dsm

        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else:
            # Labels are converted from RGB to their numeric values
            label = np.asarray(convert_from_color(io.imread(self.label_files[random_idx])), dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label

        # Get a random patch
        x1, x2, y1, y2 = get_random_pos(data, WINDOW_SIZE)
        data_p = data[:, x1:x2, y1:y2]
        dsm_p = dsm[x1:x2, y1:y2]
        # boundary_p = boundary[x1:x2, y1:y2]
        label_p = label[x1:x2, y1:y2]

        # Data augmentation
        # data_p, boundary_p, label_p = self.data_augmentation(data_p, boundary_p, label_p)
        data_p, dsm_p, label_p = self.data_augmentation(data_p, dsm_p, label_p)

        # Return the torch.Tensor values
        return (torch.from_numpy(data_p),
                torch.from_numpy(dsm_p),
                torch.from_numpy(label_p))
        
## We load one tile from the dataset and we display it
# img = io.imread('./ISPRS_dataset/Vaihingen/top/top_mosaic_09cm_area11.tif')
# fig = plt.figure()
# fig.add_subplot(121)
# plt.imshow(img)
#
# # We load the ground truth
# gt = io.imread('./ISPRS_dataset/Vaihingen/gts_for_participants/top_mosaic_09cm_area11.tif')
# fig.add_subplot(122)
# plt.imshow(gt)
# plt.show()
#
# # We also check that we can convert the ground truth into an array format
# array_gt = convert_from_color(gt)
# print("Ground truth in numerical format has shape ({},{}) : \n".format(*array_gt.shape[:2]), array_gt)


# Utils

def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


def CrossEntropy2d(input, target, weight=None, size_average=True):
    """ 2D version of the cross entropy loss """
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight, size_average)
    elif dim == 4:
        output = input.view(input.size(0), input.size(1), -1)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        return F.cross_entropy(output, target, weight, size_average)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))


def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size


def sliding_window(top, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c


def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def metrics(predictions, gts, label_values=LABELS):
    cm = confusion_matrix(
        gts,
        predictions,
        labels=range(len(label_values)))

    print("Confusion matrix :")
    print(cm)
    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("%d pixels processed" % (total))
    print("Total accuracy : %.2f" % (accuracy))

    Acc = np.diag(cm) / cm.sum(axis=1)
    for l_id, score in enumerate(Acc):
        print("%s: %.4f" % (label_values[l_id], score))
    print("---")

    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("%s: %.4f" % (label_values[l_id], score))
    print('mean F1Score: %.4f' % (np.nanmean(F1Score[:5])))
    print("---")

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    print("Kappa: %.4f" %(kappa))

    # Compute MIoU coefficient
    MIoU = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    print(MIoU)
    MIoU = np.nanmean(MIoU[:5])
    print('mean MIoU: %.4f' % (MIoU))
    print("---")

    return accuracy, MIoU
