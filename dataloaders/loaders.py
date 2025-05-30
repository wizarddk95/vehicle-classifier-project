import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from timm.data import create_transform
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

# 입력 사이즈 정의
input_sizes = {
    'resnet50': 224,
    'efficientnet_b4': 380,
    'vit_base_patch16_224': 224,
    'swin_tiny_patch4_window7_224': 224
}

# 제외할 모델
exclude = {'resnet50', 'efficientnet_b4', 'vit_base_patch16_224'}

# ✅ Subset에 transform을 적용할 수 있도록 커스텀 Dataset 클래스 정의
class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform, return_path=True):
        self.subset = subset
        self.transform = transform
        self.return_path = return_path

        if hasattr(subset, 'dataset') and hasattr(subset, 'indices'):
            self.dataset = subset.dataset
            self.indices = subset.indices
        else:
            self.dataset = subset
            self.indices = list(range(len(subset)))

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        image = self.transform(image)

        if self.return_path:
            real_index = self.indices[idx]
            image_path = self.dataset.samples[real_index][0]
            return image, label, image_path
        else:
            return image, label

# ✅ Stratified split 함수
def stratified_split(dataset, val_ratio=0.2, seed=28):
    targets = [sample[1] for sample in dataset.samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(sss.split([0] * len(targets), targets))
    print('계층적 분할 확인')
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

# ✅ ViT 전용 transform
def get_vit_transform(is_training):
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4))], p=0.8),
            # transforms.RandomApply([transforms.RandomRotation(degrees=10)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random')
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

# ✅ Swin 전용 transform
def get_swin_transform(is_training):
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4))], p=0.8),
            # transforms.RandomApply([transforms.RandomRotation(degrees=10)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random')
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

# ✅ Main 함수
def get_dataloaders(data_root, batch_size=32, val_ratio=0.2, seed=28, return_path=True):
    base_dataset = datasets.ImageFolder(root=data_root)
    model_names = [k for k in input_sizes.keys() if k not in exclude]

    # ✅ Stratified split
    if val_ratio == 0.0:
        train_subset = base_dataset
        val_subset = None
    else:
        train_subset, val_subset = stratified_split(base_dataset, val_ratio, seed)

    dataloaders = {}
    for model_name in model_names:
        input_size = input_sizes[model_name]

        if 'vit' in model_name.lower():
            train_transform = get_vit_transform(is_training=True)
            val_transform = get_vit_transform(is_training=False)
        elif 'swin' in model_name.lower():
            train_transform = get_swin_transform(is_training=True)
            val_transform = get_swin_transform(is_training=False)
        else:
            train_transform = create_transform((3, input_size, input_size), is_training=True)
            val_transform = create_transform((3, input_size, input_size), is_training=False)

        train_dataset = TransformSubset(train_subset, train_transform, return_path=return_path)
        val_dataset = TransformSubset(val_subset, val_transform, return_path=return_path) if val_subset else None

        dataloaders[model_name] = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None,
            'classes': base_dataset.classes
        }

    return dataloaders
