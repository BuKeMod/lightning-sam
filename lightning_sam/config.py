from box import Box
print('num_devices : ')
num_devices = input()

print('batch_size : ')
batch_size = input()

print('num_workers : ')
num_workers = input()

print('num_epochs : ')
num_epochs = input()

print('eval_interval : ')
eval_interval = input()

config = {
    "num_devices": num_devices,
    "batch_size": 12,
    "num_workers": 4,
    "num_epochs": 20,
    "eval_interval": 2,
    "out_dir": "/kaggle/working/training",
    "opt": {
        "learning_rate": 8e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": 'vit_b',
        "checkpoint": "/kaggle/working/sam_vit_b_qkv.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },
    "dataset": {
        "train": {
            "root_dir": "/kaggle/working/Crop-Fields-LOD-13-14-15-4/train",
            "annotation_file": "/kaggle/working/Crop-Fields-LOD-13-14-15-4/train/sa_Tannotationscoco.json"
        },
        "val": {
            "root_dir": "/kaggle/working/Crop-Fields-LOD-13-14-15-4/valid",
            "annotation_file": "/kaggle/working/Crop-Fields-LOD-13-14-15-4/valid/sa_Vannotationscoco.json"
        }
    }
}

cfg = Box(config)
