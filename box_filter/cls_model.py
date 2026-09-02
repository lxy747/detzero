import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from dataset import BFDataset
import argparse

weight_path = os.path.split(__file__)[0]


import pdb
# -------------------------- 1. 配置参数（强化预训练相关配置）--------------------------
class Config:
    # 数据配置
    data_root = "./dataset"  # 数据集根目录（train/val 子目录）
    extra_path = 'pkls_motovis_demotion'
    ann_file = ''
    val_ann_file="/data/qh_projects/DetZero-main/output_ontime/infer/epoch_13/val/infer_result.pkl"
    class_names=["car", "truck", "construction_vehicle", "bus", "bicycle", "tricycle", "pedestrian", "barrier", "traffic_cone"]
    dst_file=None,

    input_size = (96, 96)  # 输入图像分辨率
    batch_size = 32
    num_workers = 4

    # 模型与预训练配置
    backbone = "efficientnet_v2_s"  # efficientnet_v2_s
    load_official_pretrain = True  # 是否加载官方ImageNet预训练权重
    custom_pretrain_path = os.path.join(weight_path, 'weights', 'efficientnet_v2_s-dd5fe13b.pth') # 自定义预训练权重路径（优先于官方）
    ckpt=None,
    # 训练配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 10
    lr = 1e-3  # 冻结时用小学习率，解冻后可适当增大
    weight_decay = 1e-4
    patience = 3
    save_path = "weights"

config = Config()

# -------------------------- 2. 数据加载（保持不变）--------------------------

def get_dataloaders():
    test_dataset=BFDataset(data_root=config.data_root, ann_file=config.val_ann_file,
                            class_names=config.class_names, input_size=config.input_size,
                            is_training=False, is_val=True
                            )
    

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=BFDataset.collate_fn
    )
    

    train_dataset=BFDataset(data_root=config.data_root, ann_file=config.ann_file,
                            class_names=config.class_names, input_size=config.input_size,
                            is_training=True,
                            )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=BFDataset.collate_fn
    )
    return train_loader, test_loader, train_dataset.classes

# -------------------------- 3. 模型定义与预训练权重加载（核心优化）--------------------------
class ImageClassificationModel(nn.Module):
    def __init__(self, num_classes, backbone_name, load_official_pretrain, custom_pretrain_path):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name
        self.load_official_pretrain=load_official_pretrain
        self.custom_pretrain_path = custom_pretrain_path

        # 1. 加载Backbone（含官方预训练权重）
        self.backbone = self._load_backbone()

        # 2. 替换分类头（适配自定义类别数）
        self.in_features = self.backbone.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.in_features, num_classes),
        )
        self.backbone.classifier = self.classifier

        # # 3. 加载自定义预训练权重（如果存在，覆盖官方权重）
        # self._load_custom_pretrain()

    def _load_backbone(self):
        """加载Backbone并加载官方预训练权重"""
        if self.backbone_name == "efficientnet_v2_mini":
            assert False
            # weights = EfficientNet_V2_Mini_Weights.DEFAULT if self.load_official_pretrain else None
            # backbone = models.efficientnet_v2_mini(weights=weights)
        elif self.backbone_name == "efficientnet_v2_s":
            backbone = models.efficientnet_v2_s()
            if self.load_official_pretrain:
                self._load_official_pretrain(backbone)
                print(f"成功加载官方{self.backbone_name}预训练权重（ImageNet）")      
        else:
            raise ValueError(f"不支持的Backbone：{self.backbone_name}")
        return backbone

    def _load_official_pretrain(self, backbone):
        """加载自定义预训练权重（处理权重不匹配问题）"""
        if os.path.exists(self.custom_pretrain_path):
            print(f"\n加载自定义预训练权重：{self.custom_pretrain_path}")
            pretrain_weights = torch.load(self.custom_pretrain_path, map_location=config.device)
            
            # 处理权重不匹配（如分类头维度不同）
            model_weights = backbone.state_dict()
            matched_weights = {}
            unmatched_keys = []

            for key, value in pretrain_weights.items():
                # 只加载backbone的权重（分类头跳过，因为类别数可能不同）
                # pdb.set_trace()
                if value.shape == model_weights[key].shape:
                    matched_weights[key] = value
                else:
                    unmatched_keys.append(f"跳过不匹配的权重：{key}（形状：{value.shape} vs 模型形状：{model_weights.get(key, '不存在')}）")
            
            # 更新模型权重
            model_weights.update(matched_weights)
            self.load_state_dict(model_weights, strict=False)  # strict=False允许部分权重加载
            
            # 打印日志
            print(f"成功匹配并加载 {len(matched_weights)} 个权重层")
            if unmatched_keys:
                print("不匹配的权重层：")
                for msg in unmatched_keys[:5]:  # 只打印前5个
                    print(msg)
        else:
            if self.custom_pretrain_path != "./custom_pretrain.pth":  # 非默认路径才提示
                print(f"警告：自定义预训练权重路径不存在：{self.custom_pretrain_path}")

    def forward(self, x):
        return self.backbone(x)

# -------------------------- 4. 训练函数（支持冻结/解冻backbone）--------------------------
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, config:Config):
    model.to(config.device)
    best_acc = 0.0
    early_stop_count = 0
    train_log = {"loss": [], "acc": []}
    test_log = {"loss": [], "acc": []}

    for epoch in range(config.epochs):
        print(f"\n===== Epoch {epoch+1}/{config.epochs} =====")
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        pbar = tqdm(train_loader, desc="Training")
        for batch_data in pbar:

            inputs = batch_data['img']
            labels = batch_data['label'].to(torch.int64)
            # pdb.set_trace()
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            pbar.set_postfix({
                "lr": scheduler.get_last_lr()[0],
                "loss": loss.item(), 
                "acc": train_correct/train_total})
            scheduler.step()
        
        train_avg_loss = train_loss / train_total
        train_avg_acc = train_correct / train_total
        train_log["loss"].append(train_avg_loss)
        train_log["acc"].append(train_avg_acc)
        print(f"Train Loss: {train_avg_loss:.4f}, Train Acc: {train_avg_acc:.4f}")
        torch.save(model.state_dict(), config.save_path+'_ep{}.pth'.format(epoch))
        # pdb.set_trace()
        # 测试阶段
        test_avg_loss, test_avg_acc = evaluate_model(model, test_loader, criterion, config)
        test_log["loss"].append(test_avg_loss)
        test_log["acc"].append(test_avg_acc)
        print(f"Test Loss: {test_avg_loss:.4f}, Test Acc: {test_avg_acc:.4f}")

 
        # 保存最佳模型
        if test_avg_acc > best_acc:
            best_acc = test_avg_acc
            torch.save(model.state_dict(), config.save_path)
            print(f"Best model saved! Best Test Acc: {best_acc:.4f}")
            early_stop_count = 0
        else:
            early_stop_count += 1
            print(f"No improvement for {early_stop_count} epochs")

        if early_stop_count >= config.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return model

# -------------------------- 5. 评估、推理、可视化函数（保持不变）--------------------------
def evaluate_model(model, dataloader, criterion, config):
    model.to(config.device)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch_data in pbar:
            inputs = batch_data['img']
            if inputs.shape[0]==0:
                continue
            labels = batch_data['label'].to(torch.int64)

            # pdb.set_trace()
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix({"loss": loss.item(), "acc": total_correct/total_samples})

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc



def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--data_root', type=str,  required=True)
    parser.add_argument('--src_pkl', type=str, default=None, )
    parser.add_argument('--extra_path', type=str, default=None)
    parser.add_argument('--dst_pkl', type=str, default=None, required=False,)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--save_path', type=str, default=None, required=False,help='It only works when training model')
    args = parser.parse_args()
    if args.dst_pkl is None:
        args.dst_pkl = args.src_pkl[:-4] + '_bf.pkl'
        config.dst_file = args.dst_pkl
    if args.ckpt is not None:
        config.ckpt = args.ckpt
    return args

def update_cfg():
    args = parse_config()
    config.data_root = args.data_root
    config.ann_file = args.src_pkl
    config.num_workers = args.num_worker
    if args.extra_path is not None:
        config.extra_path = args.extra_path
    if args.save_path is not None:
        os.makedirs(os.path.split(args.save_path)[0], exist_ok=True)
        config.save_path = args.save_path
    

# -------------------------- 6. 主函数（执行流程）--------------------------
if __name__ == "__main__":
    update_cfg()
    
    # 1. 加载数据
    train_loader, test_loader, class_names = get_dataloaders()
    # class_names = ["car", "truck", "construction_vehicle", "bus", "bicycle", "tricycle", "pedestrian", "barrier", "traffic_cone"]
    # num_classes = len(class_names)
    # print(f"数据集类别：{class_names}，共 {num_classes} 类")

    # 2. 初始化模型（加载预训练权重）
    model = ImageClassificationModel(
        num_classes=2,
        backbone_name=config.backbone,
        load_official_pretrain=config.load_official_pretrain,
        custom_pretrain_path=config.custom_pretrain_path
    )
    # pdb.set_trace()
    print(f"\n模型初始化完成，Backbone：{config.backbone}")

    # 4. 定义优化器和调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, T_0=10, T_mult=2, eta_min=1e-6
    # )
    # 4. 初始化调度器
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.lr*10, # 0.001
        # total_steps=config.epochs * len(train_loader),
        epochs=config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=1 / config.epochs,  # 30% 迭代用于升温（≈939个迭代）
        anneal_strategy='cos',  # 余弦衰减
        div_factor=10, #25,  # base_lr = max_lr / 25 = 0.004（与优化器初始 lr 一致）
        final_div_factor=1000,  # 最终 lr = max_lr / 10000 = 1e-5
        max_momentum=0.95,
        base_momentum=0.85,
        verbose=False
    )

    # 5. 训练模型
    train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, config)

