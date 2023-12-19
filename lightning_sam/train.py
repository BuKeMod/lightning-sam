import os
import time

import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from box import Box
# from config import cfg
from dataset import load_datasets
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger
from losses import DiceLoss
from losses import FocalLoss
from model import Model
from torch.utils.data import DataLoader
from utils import AverageMeter
from utils import calc_iou

torch.set_float32_matmul_precision('high')


def validate(fabric: L.Fabric, model: Model, val_dataloader: DataLoader, epoch: int = 0):
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            images, bboxes, gt_masks = data
            num_images = images.size(0)
            pred_masks, _ = model(images, bboxes)
            for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                batch_stats = smp.metrics.get_stats(
                    pred_mask,
                    gt_mask.int(),
                    mode='binary',
                    threshold=0.5,
                )
                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                ious.update(batch_iou, num_images)
                f1_scores.update(batch_f1, num_images)
            fabric.print(
                f'Val: [{epoch}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
            )

    fabric.print(f'Validation [{epoch}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]')

    fabric.print(f"Saving checkpoint to {cfg.out_dir}")
    state_dict = model.model.state_dict()
    if fabric.global_rank == 0:
        torch.save(state_dict, os.path.join(cfg.out_dir, f"epoch-{epoch:06d}-f1{f1_scores.avg:.2f}-ckpt.pth"))
    model.train()


def train_sam(
    cfg: Box,
    fabric: L.Fabric,
    model: Model,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
):
    """The SAM training loop."""

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()

    for epoch in range(1, cfg.num_epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        focal_losses = AverageMeter()
        dice_losses = AverageMeter()
        iou_losses = AverageMeter()
        total_losses = AverageMeter()
        end = time.time()
        validated = False

        for iter, data in enumerate(train_dataloader):
            if epoch > 1 and epoch % cfg.eval_interval == 0 and not validated:
                validate(fabric, model, val_dataloader, epoch)
                validated = True

            data_time.update(time.time() - end)
            images, bboxes, gt_masks = data
            batch_size = images.size(0)
            pred_masks, iou_predictions = model(images, bboxes)
            num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
            loss_focal = torch.tensor(0., device=fabric.device)
            loss_dice = torch.tensor(0., device=fabric.device)
            loss_iou = torch.tensor(0., device=fabric.device)
            for pred_mask, gt_mask, iou_prediction in zip(pred_masks, gt_masks, iou_predictions):
                batch_iou = calc_iou(pred_mask, gt_mask)
                loss_focal += focal_loss(pred_mask, gt_mask, num_masks)
                loss_dice += dice_loss(pred_mask, gt_mask, num_masks)
                loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks

            loss_total = 20. * loss_focal + loss_dice + loss_iou
            optimizer.zero_grad()
            fabric.backward(loss_total)
            optimizer.step()
            scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()

            focal_losses.update(loss_focal.item(), batch_size)
            dice_losses.update(loss_dice.item(), batch_size)
            iou_losses.update(loss_iou.item(), batch_size)
            total_losses.update(loss_total.item(), batch_size)

            fabric.print(f'Epoch: [{epoch}][{iter+1}/{len(train_dataloader)}]'
                         f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
                         f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
                         f' | Focal Loss [{focal_losses.val:.4f} ({focal_losses.avg:.4f})]'
                         f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.avg:.4f})]'
                         f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.avg:.4f})]'
                         f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]')


def configure_opt(cfg: Box, model: Model):

    def lr_lambda(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor**2)

    optimizer = torch.optim.Adam(model.model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def main(cfg: Box) -> None:
    fabric = L.Fabric(accelerator="auto",
                      devices=cfg.num_devices,
                      strategy="auto",
                      loggers=[TensorBoardLogger(cfg.out_dir, name="lightning-sam")])
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(cfg.out_dir, exist_ok=True)

    with fabric.device:
        model = Model(cfg)
        model.setup()

    train_data, val_data = load_datasets(cfg, model.model.image_encoder.img_size)
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)

    optimizer, scheduler = configure_opt(cfg, model)
    model, optimizer = fabric.setup(model, optimizer)

    train_sam(cfg, fabric, model, optimizer, scheduler, train_data, val_data)
    validate(fabric, model, val_data, epoch=0)

######
if __name__ == "__main__":
    from box import Box
    import argparse
    
    def create_parser():
        parser = argparse.ArgumentParser(description='Your program description')

        # Training configuration
        parser.add_argument('--num_devices', type=int, default=1, help='Number of devices')
        parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
        parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
        parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
        parser.add_argument('--eval_interval', type=int, default=2, help='Evaluation interval')
        parser.add_argument('--out_dir', type=str, default='/kaggle/working/training', help='Output directory')
    
        # Optimization configuration
        parser.add_argument('--learning_rate', type=float, default=8e-4, help='Learning rate')
        parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
        parser.add_argument('--decay_factor', type=int, default=10, help='Decay factor')
        parser.add_argument('--steps', type=list, default=[60000, 86666], help='Steps')
        parser.add_argument('--warmup_steps', type=int, default=250, help='Warmup steps')
    
        # Model configuration
        parser.add_argument('--model_type', type=str, default='vit_b', help='Type of the model')
        parser.add_argument('--checkpoint', type=str, default='/kaggle/working/sam_vit_b_01ec64.pth', help='Checkpoint file path')
        parser.add_argument('--freeze_image_encoder', type=bool, default=True, help='Freeze image encoder')
        parser.add_argument('--freeze_prompt_encoder', type=bool, default=True, help='Freeze prompt encoder')
        parser.add_argument('--freeze_mask_decoder', type=bool, default=False, help='Freeze mask decoder')
    
        # Dataset configuration
        parser.add_argument('--train_root_dir', type=str, default='/kaggle/working/Crop-Fields-LOD-13-14-15-4/train/', help='Root directory for training data')
        parser.add_argument('--train_annotation_file', type=str, default='/kaggle/working/Crop-Fields-LOD-13-14-15-4/train/sa_Tannotationscoco.json', help='Annotation file for training data')
        parser.add_argument('--val_root_dir', type=str, default='/kaggle/working/Crop-Fields-LOD-13-14-15-4/valid/', help='Root directory for validation data')
        parser.add_argument('--val_annotation_file', type=str, default='/kaggle/working/Crop-Fields-LOD-13-14-15-4/valid/sa_Vannotationscoco.json', help='Annotation file for validation data')

        args = parser.parse_args()

        config = {
                "num_devices": args.num_devices,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "num_epochs": args.num_epochs,
                "eval_interval": args.eval_interval,
                "out_dir": args.out_dir,
                "opt": {
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "decay_factor": args.decay_factor,
                    "steps": args.steps,
                    "warmup_steps": args.warmup_steps,
                },
                "model": {
                    "type": args.model_type,
                    "checkpoint": args.checkpoint,
                    "freeze": {
                        "image_encoder": args.freeze_image_encoder,
                        "prompt_encoder": args.freeze_prompt_encoder,
                        "mask_decoder": args.freeze_mask_decoder,
                    },
                },
                "dataset": {
                    "train": {
                        "root_dir": args.train_root_dir,
                        "annotation_file": args.train_annotation_file
                    },
                    "val": {
                        "root_dir": args.val_root_dir,
                        "annotation_file": args.val_annotation_file
                    }
                }
            }

        
        return config


   
    def load_config(config_path):
        if isinstance(config_path, dict):
            # If config_path is already a dictionary, return it directly
            return Box(config_path)
        elif isinstance(config_path, str):
            # If config_path is a string, assume it's a JSON string and convert it to a dictionary
            return Box.from_json(config_path)
        else:
            raise ValueError("Invalid config_path type. It should be either a dictionary or a JSON string.")

  
    parser = create_parser()
    cfg = Box(parser)
    print('p ',parser)
    # print('\n')
    # print('\n')
    # args = parser.parse_args()
    # print('a ',args)
    # cfg = Box(vars(args))  # แปลง Namespace เป็น dictionary แล้วใช้ Box กลับไป
    # print('\n')
    # print('\n')
    # print('c ',cfg)
    
    main(cfg)

