import argparse
import os
from os.path import join, dirname, abspath, expanduser
from copy import deepcopy
import datetime as dt
import warnings
import sys
import random

import pickle as pkl
import yaml
from torch.utils.data import DataLoader

from deeplearning.train import train_one_epoch
from deeplearning.train import validate, final_validate

from tools import get_exp_name, copy_code, log
from tools import load_pre_pretrained_model
from tools import get_device, get_rootpath_2_dataset, create_folders_for_exp
from tools import get_yaml_args, plot_curve
from tools import get_cpu_device, announce_msg
from tools import get_transforms_tensor, get_train_transforms_img, plot_stats
from tools import check_if_allow_multgpu_mode
from tools import copy_model_state_dict_from_gpu_to_cpu

from loader import csv_loader, PhotoDataset, default_collate, MyDataParallel

from instantiators import instantiate_models, instantiate_optimizer
from instantiators import instantiate_loss

import torch

import reproducibility
from reproducibility import reset_seed, set_default_seed

CONST1 = 1000  # used to generate random numbers.


# args.num_workers * this_factor. Useful when setting set_for_eval to False,
# batch size =1.
FACTOR_MUL_WORKERS = 2
# and we are in an evaluation mode (to go faster and coop with the lag
# between the CPU and GPU).

# Can be activated only for "Caltech-UCSD-Birds-200-2011" or
# "Oxford-flowers-102"
DEBUG_MODE = False
# dataset to go fast. If True, we select only few samples for training
# , validation, and test.
PLOT_STATS = True

# use the default seed. Copy the see into the os.environ("MYSEED")
reproducibility.init_seed()

NBRGPUS = torch.cuda.device_count()

ALLOW_MULTIGPUS = check_if_allow_multgpu_mode()


def _init_fn(worker_id):
    """
    Init. function for the worker in dataloader.
    :param worker_id:
    :return:
    """
    pass  # no longer necessary since we override the process seed.


if __name__ == "__main__":

    # =============================================
    # Parse the inputs and deal with the yaml file.
    # =============================================

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml", type=str, help="yaml file containing the configuration.")
    parser.add_argument("--cudaid", type=str, default="0", help="cuda id.")
    input_args, _ = parser.parse_known_args()
    args, args_dict = get_yaml_args(input_args)

    # ===============
    # Reproducibility
    # ===============

    # ==================================================
    # Device, criteria, folders, output logs, callbacks.
    # ==================================================

    DEVICE = get_device(args)
    CPUDEVICE = get_cpu_device()

    CRITERION = instantiate_loss(args).to(DEVICE)

    # Write in scratch instead of /project
    if "CC_CLUSTER" in os.environ.keys():
        FOLDER = '{}'.format(os.environ["SCRATCH"])
        OUTD = join(
            FOLDER, "fixed-weighted-{}-{}-{}-{}-{}-exps-unimodal-const-"
                    "new-selection-criteria-mxepochs-{}".format(
                        args.weight_ce, args.loss, args.dataset,
                        args.max_epochs, args.mulcoef, args.max_epochs),
            get_exp_name(args))
    else:
        # we need to write in home...
        # parent_folder = dirname(abspath(__file__)).split(os.sep)[-1]
        # FOLDER = join("{}/code".format(os.environ["NEWHOME"]), parent_folder)
        # OUTD = join(FOLDER, "exps", get_exp_name(args))
        # TODO: fix the name of the exp so to get it all from get_exp_name on
        #  cc and anywhere else.
        OUTD = join(dirname(abspath(__file__)), "exps",
                    "{}-{}-{}-{}".format(
                        args.loss, args.dataset, args.max_epochs,
                        get_exp_name(args)))
        OUTD = expanduser(OUTD)

    if not os.path.exists(OUTD):
        os.makedirs(OUTD)

    OUTD_TR = create_folders_for_exp(OUTD, "train")
    OUTD_VL = create_folders_for_exp(OUTD, "validation")
    OUTD_TS = create_folders_for_exp(OUTD, "test")
    OUTD_TLB = create_folders_for_exp(OUTD, "tlb")

    subdirs = ["init_params"]
    for sbdr in subdirs:
        if not os.path.exists(join(OUTD, sbdr)):
            os.makedirs(join(OUTD, sbdr))

    # save the yaml file.
    if not os.path.exists(join(OUTD, "code/")):
        os.makedirs(join(OUTD, "code/"))
    with open(join(OUTD, "code/", input_args.yaml), 'w') as fyaml:
        yaml.dump(args_dict, fyaml)

    copy_code(join(OUTD, "code/"))

    training_log = join(OUTD, "training.txt")
    results_log = join(OUTD, "results.txt")

    log(training_log, "\n\n ########### Training #########\n\n")
    log(results_log, "\n\n ########### Results #########\n\n")

    callback = None

    # ==========================================================
    # Data transformations: on PIL.Image.Image and torch.tensor.
    # ==========================================================

    train_transform_img = get_train_transforms_img(args)
    transform_tensor = get_transforms_tensor(args)

    # ==========================================================================
    # Datasets: load csv, datasets: train, valid, test.
    # ==========================================================================

    announce_msg("SPLIT: {} \t FOLD: {}".format(args.split, args.fold))

    relative_fold_path = join(
        args.fold_folder, args.dataset, "split_" + str(args.split),
        "fold_" + str(args.fold)
    )
    if isinstance(args.name_classes, str):  # path
        path_classes = join(relative_fold_path, args.name_classes)
        assert os.path.isfile(path_classes), "File {} does not exist .... " \
                                             "[NOT OK]".format(path_classes)
        with open(path_classes, "r") as fin:
            args.name_classes = yaml.load(fin)
    csvfiles = []
    for subp in ["train_s_", "valid_s_", "test_s_"]:
        csvfiles.append(
            join(
                relative_fold_path,
                subp + str(args.split) + "_f_" + str(args.fold) + ".csv")
        )
    train_csv, valid_csv, test_csv = csvfiles
    # Check if the csv files exist. If not, raise an error.
    if not all([os.path.isfile(train_csv), os.path.isfile(valid_csv),
                os.path.isfile(test_csv)]):
        raise ValueError(
            "Missing *.cvs files ({}[{}], {}[{}], {}[{}])".format(
                train_csv, os.path.isfile(train_csv), valid_csv,
                os.path.isfile(valid_csv), test_csv, os.path.isfile(test_csv)))

    rootpath = get_rootpath_2_dataset(args)

    train_samples = csv_loader(train_csv, rootpath)
    valid_samples = csv_loader(valid_csv, rootpath)
    test_samples = csv_loader(test_csv, rootpath)

    # Just for debug to go fast.
    if DEBUG_MODE:
        set_default_seed()
        warnings.warn("YOU ARE IN DEBUG MODE!!!!")
        if args.dataset == "Caltech-UCSD-Birds-200-2011":
            nbrx_tr, nbrx_vl, nbrx_tst = 100, 5, 20
        elif args.dataset == "Oxford-flowers-102":
            nbrx_tr, nbrx_vl, nbrx_tst = 100, 5, 20
        elif args.dataset == "glas":
            nbrx_tr, nbrx_vl, nbrx_tst = 20, 5, 20
        elif args.dataset =="bach-part-a-2018":
            nbrx_tr, nbrx_vl, nbrx_tst = 20, 5, 20
        elif args.dataset =="fgnet":
            nbrx_tr, nbrx_vl, nbrx_tst = 20, 5, 20
        elif args.dataset =="afad-lite":
            nbrx_tr, nbrx_vl, nbrx_tst = 200, 5000, 200
        elif args.dataset =="afad-full":
            nbrx_tr, nbrx_vl, nbrx_tst = 200, 5000, 200
        elif args.dataset =="historical-color-image-decade":
            nbrx_tr, nbrx_vl, nbrx_tst = 20, 5, 20
        else:
            raise ValueError("Unknown dataset: {} ....[NOT OK]")
        train_samples = random.sample(train_samples, nbrx_tr)
        valid_samples = random.sample(valid_samples, nbrx_vl)
        test_samples = test_samples[:nbrx_tst]
        set_default_seed()

    announce_msg("creating datasets and dataloaders")

    set_default_seed()
    trainset = PhotoDataset(
        train_samples, args.dataset, args.name_classes, transform_tensor,
        set_for_eval=False, transform_img=train_transform_img,
        resize=args.resize, crop_size=args.crop_size,
        padding_size=args.padding_size, padding_mode=args.padding_mode,
        up_scale_small_dim_to=args.up_scale_small_dim_to,
        do_not_save_samples=True,
        ratio_scale_patch=args.ratio_scale_patch
    )

    # TODO: find a better way to protect the reproducibility of operations that
    #  changes any random generator's state.
    #  We will call it: reproducibility armor. Functions/classes/operations
    #  that use a random generator should be independent from each other.
    set_default_seed()
    validset = PhotoDataset(
        valid_samples, args.dataset, args.name_classes, transform_tensor,
        set_for_eval=False, transform_img=None, resize=args.resize,
        crop_size=None,
        padding_size=None if not args.pad_eval else args.padding_size,
        padding_mode=None if not args.pad_eval else args.padding_mode,
        up_scale_small_dim_to=args.up_scale_small_dim_to,
        do_not_save_samples=True,
        ratio_scale_patch=args.ratio_scale_patch, for_eval_flag=True
    )

    set_default_seed()
    train_loader = DataLoader(trainset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, worker_init_fn=_init_fn,
                              collate_fn=default_collate)

    set_default_seed()
    # we need more workers since the batch size is 1, and set_for_eval is
    # False (need more time to prepare a sample).
    valid_loader = DataLoader(
        validset, batch_size=1, shuffle=False,
        num_workers=args.num_workers * FACTOR_MUL_WORKERS, pin_memory=True,
        collate_fn=default_collate, worker_init_fn=_init_fn
    )
    set_default_seed()

    # ################## Instantiate models ######################
    model = instantiate_models(args)

    # Check if we are using a user specific pre-trained model other than our
    # pre-defined pre-trained models. This can be used to to EVALUATE a
    # trained model. You need to set args.max_epochs to -1 so no training is
    # performed. This is a hack to avoid creating other function to deal with
    # LATER-evaluation after this code is done. This script is intended for
    # training. We evaluate at the end. However, if you missed something
    # during the training/evaluation (for example plot something over the
    # predicted images), you do not need to re-train the model. You can 1.
    # specify the path to the pre-trained model. 2. Set max_epochs to -1. 3.
    # Set strict to True. By doing this, we load the pre-trained model, and,
    # we skip the training loop, fast-forward to the evaluation.

    if hasattr(args, "path_pre_trained"):
        warnings.warn("You have asked to load a specific pre-trained "
                      "model from {} .... [OK]".format(args.path_pre_trained))
        model = load_pre_pretrained_model(
            model=model, path_file=args.path_pre_trained, strict=args.strict)

    # Check if there are multiple GPUS.
    if ALLOW_MULTIGPUS:
        model = MyDataParallel(model)
        if args.batch_size < NBRGPUS:
            warnings.warn("You asked for MULTIGPU mode. However, "
                          "your batch size {} is smaller than the number of "
                          "GPUs available {}. This is fine in practice. "
                          "However, some GPUs will be idol. "
                          "This is just a warning .... "
                          "[OK]".format(args.batch_size, NBRGPUS))
    model.to(DEVICE)
    # Copy the model's params.
    best_state_dict = deepcopy(model.state_dict())  # it has to be deepcopy.

    # ############################### Instantiate optimizer ####################
    set_default_seed()
    optimizer, lr_scheduler = instantiate_optimizer(args, model)

    # ################################ Training ################################
    set_default_seed()
    tr_stats, vl_stats = None, None

    best_val_metric = None
    best_epoch = 0

    # TODO: validate before start training.
    # reset_seed(int(os.environ["MYSEED"]))
    # vl_stats = validate(
    #     model, valid_loader, CRITERION, DEVICE, vl_stats, -1,
    #     training_log)
    # reset_seed(int(os.environ["MYSEED"]))

    announce_msg("start training")
    set_default_seed()
    tx0 = dt.datetime.now()

    for epoch in range(args.max_epochs):
        # reseeding tr/vl samples.
        reset_seed(int(os.environ["MYSEED"]) + (epoch + 1) * CONST1)
        trainset.set_up_new_seeds()
        reset_seed(int(os.environ["MYSEED"]) + (epoch + 2) * CONST1)
        validset.set_up_new_seeds()

        # Start the training with fresh seeds.
        reset_seed(int(os.environ["MYSEED"]) + (epoch + 3) * CONST1)

        tr_stats = train_one_epoch(
            model, optimizer, train_loader, CRITERION, DEVICE, tr_stats, epoch,
            training_log, ALLOW_MULTIGPUS=ALLOW_MULTIGPUS, NBRGPUS=NBRGPUS)

        if lr_scheduler:  # for Pytorch > 1.1 : opt.step() then l_r_s.step().
            lr_scheduler.step(epoch)
        # TODO: Which criterion to use over the validation set?
        #  (in case this loss is used for model selection).
        # Eval validation set.
        reset_seed(int(os.environ["MYSEED"]) + (epoch + 4) * CONST1)
        vl_stats = validate(
            model, valid_loader, CRITERION, DEVICE, vl_stats, epoch,
            training_log)

        reset_seed(int(os.environ["MYSEED"]) + (epoch + 5) * CONST1)

        # validation metrics: acc, mae, soi_y, soi_py, loss
        acc, mae, soi_y, soi_py, vl_loss = vl_stats[-1, :]
        metric_val = vl_loss

        if args.loss in ["LossPN", "LossELB", "LossRLB"]:
            # in our case, we should not use the loss as a selection criterion
            # since it combines two terms.
            if args.dataset in ["bach-part-a-2018",
                                "historical-color-image-decade"]:
                metric_val = 1. - acc
            else:
                metric_val = mae

        # TODO: Revise the model selection: acc.
        if (best_val_metric is None) or (metric_val <= best_val_metric):
            # best_val_loss:
            best_val_metric = metric_val
            # it has to be deepcopy.
            best_state_dict = deepcopy(model.state_dict())
            # Expensive operation: disc I/O.
            # torch.save(best_model.state_dict(), join(OUTD, "best_model.pt"))
            best_epoch = epoch

        if PLOT_STATS:
            print("Plotting stats ...")
            plot_stats(tr_stats, join(OUTD_TR.folder, "trainset-stats.png"),
                       title="Trainset stats. {}".format(args.loss), dpi=100,
                       plot_avg=True, moving_avg=30)
            plot_stats(vl_stats, join(OUTD_VL.folder, "validset-stats.png"),
                       title="Validset stats. {}".format(args.loss), dpi=100,
                       plot_avg=True, moving_avg=30)

        CRITERION.update_t()
        if CRITERION.t_tracker:
            plot_curve(
                CRITERION.t_tracker,
                join(OUTD_TLB.folder, "tlb-evolution.png"),
                "t evolution. min: {}. max: {}.".format(
                    min(CRITERION.t_tracker), max(CRITERION.t_tracker)),
                "epochs", "t", dpi=100
            )
    # ==========================================================================
    #                                   DO CLOSING-STUFF BEFORE LEAVING
    # ==========================================================================
    # Classification errors using the best model over: train/valid/test sets.
    # Train set: needs to reload it with eval-transformations,
    # not train-transformations.

    # Reset the models parameters to the best found ones.
    model.load_state_dict(best_state_dict)

    log(
        results_log, "Loss: {} \n Best epoch: {}".format(best_epoch, args.loss))

    # We need to do each set sequentially to free the memory.
    msg = "End training. Time: {}".format(dt.datetime.now() - tx0)

    announce_msg(msg)
    log(training_log, msg)

    if CRITERION.t_tracker:
        with open(join(OUTD_TLB.folder, "tlb-evolution.pkl"), "wb") as ft:
            pkl.dump(CRITERION.t_tracker, ft, pkl.HIGHEST_PROTOCOL)

    # Save train statistics (train, valid)
    stats_to_dump = {
        "train": tr_stats,
        "valid": vl_stats
    }
    with open(join(OUTD, "train_stats.pkl"), "wb") as fout:
        pkl.dump(stats_to_dump, fout, protocol=pkl.HIGHEST_PROTOCOL)

    tx0 = dt.datetime.now()

    set_default_seed()
    msg = "start final processing stage \nBest epoch: {}".format(best_epoch)
    announce_msg(msg)
    log(training_log, msg)


    if DEBUG_MODE and (args.dataset in [
            "bach-part-a-2018", "fgnet", "afad-lite", "afad-full",
            "Caltech-UCSD-Birds-200-2011", "Oxford-flowers-102",
            "historical-color-image-decade"
    ]):
        set_default_seed()
        testset = PhotoDataset(
            test_samples, args.dataset, args.name_classes, transform_tensor,
            set_for_eval=False, transform_img=None, resize=args.resize,
            crop_size=None,
            padding_size=None if not args.pad_eval else args.padding_size,
            padding_mode=None if not args.pad_eval else args.padding_mode,
            up_scale_small_dim_to=args.up_scale_small_dim_to,
            do_not_save_samples=True, ratio_scale_patch=args.ratio_scale_patch,
            for_eval_flag=True
        )

        set_default_seed()
        test_loader = DataLoader(
            testset, batch_size=1, shuffle=False, num_workers=0,
            pin_memory=True,
            collate_fn=default_collate, worker_init_fn=_init_fn
        )

        set_default_seed()
        final_validate(
            model, test_loader, CRITERION, DEVICE, testset, OUTD_TS.folder,
            results_log, name_set="testset"
        )

        del testset
        del test_loader

        # Move the state dict of the best model into CPU, then save it.
        best_state_dict_cpu = copy_model_state_dict_from_gpu_to_cpu(model)
        torch.save(best_state_dict_cpu, join(OUTD, "best_model.pt"))

        msg = "End final processing. ***DEBUG MODE ON*** Time: {}".format(
            dt.datetime.now() - tx0)
        announce_msg(msg)
        log(training_log, msg)
        announce_msg("*END*")

        sys.exit()

    del trainset
    del train_loader

    # ==========================================================================
    #                               VALIDATION SET
    # ==========================================================================
    if args.dataset not in ["afad-full"]:  # it takes more than 1h.
        set_default_seed()
        final_validate(
            model, valid_loader, CRITERION, DEVICE, validset, OUTD_VL.folder,
            results_log, name_set="validset"
        )
        del validset
        del valid_loader

    # ==========================================================================
    #                               TEST SET
    # ==========================================================================
    set_default_seed()
    testset = PhotoDataset(
        test_samples, args.dataset, args.name_classes, transform_tensor,
        set_for_eval=False, transform_img=None, resize=args.resize,
        crop_size=None,
        padding_size=None if not args.pad_eval else args.padding_size,
        padding_mode=None if not args.pad_eval else args.padding_mode,
        up_scale_small_dim_to=args.up_scale_small_dim_to,
        do_not_save_samples=True, ratio_scale_patch=args.ratio_scale_patch,
        for_eval_flag=True
    )

    set_default_seed()
    test_loader = DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True,
        collate_fn=default_collate, worker_init_fn=_init_fn
    )

    set_default_seed()
    final_validate(
        model, test_loader, CRITERION, DEVICE, testset, OUTD_TS.folder,
        results_log, name_set="testset"
    )

    del testset
    del test_loader

    # ==========================================================================
    #                               TRAIN SET
    # ==========================================================================
    if args.dataset not in ["afad-full"]:  # it takes more than 1h.
        set_default_seed()
        trainset_eval = PhotoDataset(
            train_samples, args.dataset, args.name_classes, transform_tensor,
            set_for_eval=False, transform_img=None, resize=args.resize,
            crop_size=None,
            padding_size=None if not args.pad_eval else args.padding_size,
            padding_mode=None if not args.pad_eval else args.padding_mode,
            up_scale_small_dim_to=args.up_scale_small_dim_to,
            do_not_save_samples=True, ratio_scale_patch=args.ratio_scale_patch,
            for_eval_flag=True
        )

        set_default_seed()
        train_eval_loader = DataLoader(
            trainset_eval, batch_size=1, shuffle=False, num_workers=0,
            pin_memory=True, collate_fn=default_collate, worker_init_fn=_init_fn
        )

        set_default_seed()
        final_validate(
            model, train_eval_loader, CRITERION, DEVICE, trainset_eval,
            OUTD_TR.folder, results_log, name_set="trainset"
        )
        del trainset_eval
        del train_eval_loader

    # Move the state dict of the best model into CPU, then save it.
    best_state_dict_cpu = copy_model_state_dict_from_gpu_to_cpu(model)
    torch.save(model.state_dict(), join(OUTD, "best_model.pt"))

    msg = "End final processing. ***DEBUG MODE OFF*** Time: {}".format(
        dt.datetime.now() - tx0)
    announce_msg(msg)
    log(training_log, msg)

    announce_msg("*END*")
    # ==========================================================================
    #                              END
    # ==========================================================================
