#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""yaml-generate-configs.py
Generates yaml configurations for experiments.
"""
import datetime as dt
from os.path import join

import yaml


# DEFAULT configuration.
config = {
    # ######################### GENERAL STUFF ##############################
    "MYSEED": 0,  # Seed for reproducibility. int >= 0.
    "dataset": "bach-part-a-2018",  # name of the dataset:
    # ordinal:bach-part-a-2018, fgnet, afad-lite, afad-full,
    # historical-color-image-decade
    # nominal: Caltech-UCSD-Birds-200-2011, Oxford-flowers-102
    "name_classes": {'Normal': 0, 'Benign': 1, 'InSitu': 2, 'Invasive': 3},
    # dict. name classes and corresponding int. If dict if too big,
    # you can dump it in the fold folder in a yaml file.
    # We will load it when needed. Use the name of the file.
    "num_classes": 4,  # Total number of classes.
    "split": 0,  # split id.
    "fold": 0,  # folder id.
    "fold_folder": "./folds",  # relative path to the folder of the folds.
    "resize": None,  # PIL format of the image size (w, h).
    # The size to which the original images are resized to.
    "crop_size": (512, 512),  # Size of the patches to be cropped (h, w).
    "ratio_scale_patch": 0.5,  # the ratio to which the cropped patch is scaled.
    # during evaluation, original images are also rescaled using this ratio.
    # if you want to keep the cropped patch as it is, set this variable to 1.
    "up_scale_small_dim_to": None,  # int or None. Upscale only images that
    # have the min-dimension is lower than this variable.
    # If int, the images are upscaled to this size while preserving the
    # ratio. See loader.PhotoDataset().
    "padding_size": None,  # (0.5, 0.5),  # padding ratios for the original
    # image for (top/bottom) and (left/right). Can be applied on both,
    # training/evaluation modes. To be specified in PhotoDataset().
    # If specified, only training images are padded. To pad evaluation
    # images, you need to set the variable: `pad_eval` to True.
    "pad_eval": False,  # If True, evaluation images are padded in the same way.
    # The final mask is cropped inside the predicted mask (since this last one
    # is bigger due to the padding).
    "padding_mode": "reflect",  # type of padding. Accepted modes:
    # https://pytorch.org/docs/stable/torchvision/transforms.html#
    # torchvision.transforms.functional.pad
    "preload": True,  # If True, images are loaded and saved in RAM to avoid
    # disc access.
    "batch_size": 8,  # the batch size for training.
    "num_workers": 8,  # number of workers for dataloader of the trainset.
    "max_epochs": 400,  # number of training epochs.
    # ############# VISUALISATION OF REGIONS OF INTEREST #################
    "floating": 3,  # the number of floating points to print over the maps.
    "height_tag": 50,  # the height of the margin where the tag is written.
    "extension": ("jpeg", "JPEG"),  # format into which the maps are saved.
    # ######################### Optimizer ##############################
    "optimizer": {  # the optimizer
        # ==================== SGD =======================
        "name": "sgd",  # str name.
        "lr": 0.001,  # Initial learning rate.
        "momentum": 0.9,  # Momentum.
        "dampening": 0.,  # dampening.
        "weight_decay": 1e-5,  # The weight decay (L2) over the parameters.
        "nesterov": True,  # If True, Nesterov algorithm is used.
        # ==================== ADAM =========================
        # "name": "adam",  # str name.
        # "lr": 0.0001,  # Initial learning rate.
        # "betas": (0.9, 0.999),  # betas.
        # "weight_decay": 0.0005,  # The weight decay (L2) over the parameters.
        # "eps": 1e-08,  # eps. for numerical stability.
        # "amsgrad": False,  # Use amsgrad variant or not.
        # ========== LR scheduler: how to adjust the learning rate. ============
        "lr_scheduler": {
            # ========> torch.optim.lr_scheduler.StepLR
            # "name": "step",  # str name.
            # "step_size": 20,  # Frequency of which to adjust the lr.
            # "gamma": 0.1,  # the update coefficient: lr = gamma * lr.
            # "last_epoch": -1,  # the index of the last epoch where to stop
            # # adjusting the LR.
            # ========> MyStepLR: override torch.optim.lr_scheduler.StepLR
            "name": "mystep",  # str name.
            "step_size": 40,  # Frequency of which to adjust the lr.
            "gamma": 0.1,  # the update coefficient: lr = gamma * lr.
            "last_epoch": -1,  # the index of the last epoch where to stop
            # adjusting the LR.
            "min_lr": 1e-7,  # minimum allowed value for lr.
            # ========> torch.optim.lr_scheduler.MultiStepLR
            # "name": "multistep",  # str name.
            # "milestones": [0, 100],  # milestones.
            # "gamma": 0.1,  # the update coefficient: lr = gamma * lr.
            # "last_epoch": -1  # the index of the last epoch where to stop
            # # adjusting the LR.
        }
    },
    # ######################### Model ##############################
    "model": {
        "name": "resnet18",  # name of the classifier.
        "pretrained": True,  # use/or not the ImageNet pretrained models.
        # =============================  classifier ==========================
        "modalities": 5,  # number of modalities (wildcat).
        "kmax": 0.1,  # kmax. (wildcat)
        "kmin": 0.1,  # kmin. (wildcat)
        "alpha": 0.0,  # alpha. (wildcat)
        "dropout": 0.0,  # dropout over the kmin and kmax selected activations
        # .. (wildcat).
    },
    # ######################### loss ##############################
    "loss": "LossCE",  # "LossCE", "LossPN", "LossELB", "LossRLB", "LossREN",
    # "LossLD", "LossMV", "LossPO".
    "lamb": 1e-5,  # lambda: LossPN
    "eps": 1e-1,  # epsilon: LossPN
    "init_t": 1.,  # LossELB, LossRLB
    "max_t": 10.,  # LossELB, LossRLB
    "mulcoef": 1.01,  # LossELB, LossRLB
    "epsp": 1e-1,  # LossRLB
    "thrs": 0.5,  # LossREN
    "var": 1.,  # LossLD
    "lam1": 0.2,  # LossMV
    "lam2": 0.05,  # LossMV
    "tau": 1.,  # LossPO
    "weight_ce": False  # ELB, RLB. if true, the CE loss is weighted by t.
}

# ==============================================================================
#                          PUBLIC LOSSES
#                 1. LossCE: Cross-entropy loss.
#                 2. LossPN: Penalty-based loss.
#                 3. LossELB: Extended log-barrier loss.
#                 4. LossRLB: Rectified log-barrier loss.
#                 5. LossREN: Re-encode loss.
#                 6. LossLD: Label distribution loss.
#                 7. LossMV: Mean-variance loss.
#                 8. LossPO: Hard-wired Poisson loss.
# ==============================================================================

# bach-part-a-2018: 4 classes.
# fgnet: 70 classes.
# afad-lite: 22 classes.
# afad-full: 58 classes.
# historical-color-image-decade: 5

l_datasets = [
    "bach-part-a-2018",
    "fgnet",
    "afad-lite",
    "afad-full",
    "historical-color-image-decade"]

for ds in l_datasets:
    config.update({"dataset": ds})
    # rectify depending on the dataset.
    if config["dataset"] == "fgnet":
        config["num_classes"] = 70
        config["name_classes"] = "encoding.yaml"
        # stats images
        # min h 359, 	 max h 772
        # min w 300, 	 max w 639
        config["up_scale_small_dim_to"] = 532
    elif config["dataset"] == "afad-lite":
        config["num_classes"] = 22
        config["name_classes"] = "encoding.yaml"
        # stats images
        # min h 100, 	 max h 536
        # min w 100, 	 max w 536
        config["up_scale_small_dim_to"] = 532
    elif config["dataset"] == "afad-full":
        config["num_classes"] = 58
        config["name_classes"] = "encoding.yaml"
        # stats images
        # min h 20, 	 max h 536
        # min w 20, 	 max w 536
        config["up_scale_small_dim_to"] = 532
    elif config["dataset"] == "historical-color-image-decade":
        config["num_classes"] = 5
        config["name_classes"] = "encoding.yaml"
        # stats images
        # min h 312, 	 max h 500
        # min w 312, 	 max w 500
        config["up_scale_small_dim_to"] = None
        config["crop_size"] = (256, 256)  # use small patches since the
        # content is not really important but the colors.
        config["ratio_scale_patch"] = 1.
    fold_yaml = "config_yaml"
    fold_bash = "config_bash"
    # name_config = dt.datetime.now().strftime('%m_%d_%Y_%H_%M_%S_%f')
    name_config = config['dataset']
    name_bash = join(fold_bash, name_config + ".sh")
    name_yaml = join(fold_yaml, name_config + ".yaml")

    with open(name_yaml, 'w') as f:
        yaml.dump(config, f)
