import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from cls_model import config, update_cfg, ImageClassificationModel
from dataset import BFDataset
import pdb

def infer_model(model, dataloader, config):
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

def main():
    update_cfg()
    torch.seed()
    test_dataset=BFDataset(data_root=config.data_root, ann_file=config.ann_file,
                           extra_path=config.extra_path,
                            class_names=config.class_names, input_size=config.input_size,
                            is_training=False, 
                            is_val=False,
                            expand_ratio=0.0,
                            do_key_frame=True
                            )
    

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=BFDataset.collate_fn
    )

    model = ImageClassificationModel(
        num_classes=2,
        backbone_name=config.backbone,
        load_official_pretrain=False, # config.load_official_pretrain,
        custom_pretrain_path=config.custom_pretrain_path
    )

    ckpt = config.ckpt
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt))
    else:
        assert False, "ckpt {} is invalid".format(ckpt)
    model.eval()
    model.to(config.device)
    for i, batch_data in tqdm(enumerate(test_loader), desc='infer boxes filter', total=len(test_loader)):
        sample_id = batch_data['sample_id']
        ids_in_3d = batch_data['ids_in_3d']
        inputs = batch_data['img']
        if inputs.shape[0]==0:
            continue
        labels = batch_data['label'].to(torch.int64)
        # pdb.set_trace()
        inputs, labels = inputs.to(config.device), labels.to(config.device)
        outputs = model(inputs)
        outputs = torch.softmax(outputs, dim=1).cpu()
        # pdb.set_trace()
        test_dataset.deduplication_boxes3d(outputs, sample_id, ids_in_3d)
    
    test_dataset.save_ann(config.dst_file)
  
   
   
  

if __name__ =='__main__':
    main()