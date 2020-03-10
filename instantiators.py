from collections import Sequence
import warnings

from torch.optim import SGD
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler


from deeplearning import models, criteria
from deeplearning import lr_scheduler as my_lr_scheduler
from tools import Dict2Obj, count_nb_params
import loader


def instantiate_loss(args):
    """
    Instantiate the train loss.
    Can be used as well for evaluation.

    :param args: Dict2Obj. Contains the configuration of the exp that has been
    read from the yaml file.
    :return: train_loss: instance of deeplearning.criteria.py losses.
    """
    loss_str = args.loss
    msg = "Unknown loss {}. It must be one of " \
          "[`LossCE`, `LossPN`, `LossELB`, `LossRLB`, `LossREN`, `LossLD`," \
          "`LossMV`, `LossPO`] ... [NOT OK]".format(loss_str)

    assert loss_str in [
        "LossCE", "LossPN", "LossELB", "LossRLB", "LossREN", "LossLD",
        "LossMV", "LossPO"], msg
    if loss_str == "LossCE":
        return criteria.__dict__[args.loss]()
    if loss_str == "LossPN":
        return criteria.__dict__[args.loss](lamb=args.lamb, eps=args.eps)
    if loss_str == "LossELB":
        return criteria.__dict__[args.loss](
            init_t=args.init_t, max_t=args.max_t, mulcoef=args.mulcoef,
            weight_ce=args.weight_ce)
    if loss_str == "LossRLB":
        return criteria.__dict__[args.loss](
            init_t=args.init_t, max_t=args.max_t, mulcoef=args.mulcoef,
            epsp=args.epsp, weight_ce=args.weight_ce)
    if loss_str == "LossREN":
        return criteria.__dict__[args.loss](thrs=args.thrs)
    if loss_str == "LossLD":
        return criteria.__dict__[args.loss](var=args.var)
    if loss_str == "LossMV":
        return criteria.__dict__[args.loss](lam1=args.lam1, lam2=args.lam2)
    if loss_str == "LossPO":
        return criteria.__dict__[args.loss]()

    raise ValueError("Something's wrong. We do not even know how you "
                     "reached this line. Loss_str: {}"
                     " .... [NOT OK]. EXITING".format(loss_str))


def instantiate_models(args):
    """Instantiate the necessary models.
    Input:
        args: Dict2Obj. Contains the configuration of the exp that has been read
        from the yaml file.

    Output:
        instance of a model.
    """
    p = Dict2Obj(args.model)
    model = models.__dict__[p.name](pretrained=p.pretrained,
                                    num_classes=args.num_classes,
                                    modalities=p.modalities, kmax=p.kmax,
                                    kmin=p.kmin, alpha=p.alpha,
                                    dropout=p.dropout,
                                    poisson=(args.loss == "LossPO"),
                                    tau=args.tau)

    print("`{}` was successfully instantiated. Nbr.params: {} .... [OK]".format(
        model, count_nb_params(model)))
    return model


def instantiate_optimizer(args, model):
    """Instantiate an optimizer.
    Input:
        args: object. Contains the configuration of the exp that has been
        read from the yaml file.
        mode: a pytorch model with parameters.

    Output:
        optimizer: a pytorch optimizer.
        lrate_scheduler: a pytorch learning rate scheduler (or None).
    """
    if args.optimizer["name"] == "sgd":
        optimizer = SGD(model.parameters(), lr=args.optimizer["lr"],
                        momentum=args.optimizer["momentum"],
                        dampening=args.optimizer["dampening"],
                        weight_decay=args.optimizer["weight_decay"],
                        nesterov=args.optimizer["nesterov"])
    elif args.optimizer["name"] == "adam":
        optimizer = Adam(params=model.parameters(), lr=args.optimizer["lr"],
                         betas=args.optimizer["betas"],
                         eps=args.optimizer["eps"],
                         weight_decay=args.optimizer["weight_decay"],
                         amsgrad=args.optimizer["amsgrad"])
    else:
        raise ValueError("Unsupported optimizer `{}` .... "
                         "[NOT OK]".format(args.optimizer["name"]))

    print("Optimizer `{}` was successfully instantiated .... "
          "[OK]".format(
            [key + ":" + str(args.optimizer[key]) for
                key in args.optimizer.keys()]))

    if args.optimizer["lr_scheduler"]:
        if args.optimizer["lr_scheduler"]["name"] == "step":
            lr_scheduler_ = args.optimizer["lr_scheduler"]
            lrate_scheduler = lr_scheduler.StepLR(
                optimizer, step_size=lr_scheduler_["step_size"],
                gamma=lr_scheduler_["gamma"],
                last_epoch=lr_scheduler_["last_epoch"])
            print(
                "Learning scheduler `{}` was successfully instantiated "
                ".... [OK]".format(
                    [key + ":" + str(lr_scheduler_[key]) for
                     key in lr_scheduler_.keys()]))
        elif args.optimizer["lr_scheduler"]["name"] == "mystep":
            lr_scheduler_ = args.optimizer["lr_scheduler"]
            lrate_scheduler = my_lr_scheduler.MyStepLR(
                optimizer, step_size=lr_scheduler_["step_size"],
                gamma=lr_scheduler_["gamma"],
                last_epoch=lr_scheduler_["last_epoch"],
                min_lr=lr_scheduler_["min_lr"])
            print("Learning scheduler `{}` was successfully instantiated ...."
                  " [OK]".format([key + ":" + str(lr_scheduler_[key]) for
                                  key in lr_scheduler_.keys()]))
        elif args.optimizer["lr_scheduler"]["name"] == "multistep":
            lr_scheduler_ = args.optimizer["lr_scheduler"]
            lrate_scheduler = lr_scheduler.MultiStepLR(
                optimizer, milestones=lr_scheduler_["milestones"],
                gamma=lr_scheduler_["gamma"],
                last_epoch=lr_scheduler_["last_epoch"])
            print(
                "Learning scheduler `{}` was successfully instantiated"
                " .... [OK]".format(
                    [key + ":" + str(lr_scheduler_[key]) for
                     key in lr_scheduler_.keys()]))
        else:
            raise ValueError("Unsupported learning rate scheduler `{}` .... "
                             "[NOT OK]".format(
                                args.optimizer["lr_scheduler"]["name"]))
    else:
        lrate_scheduler = None

    return optimizer, lrate_scheduler



def instantiate_patch_splitter(args, deterministic=True):
    """
    Instantiate the patch splitter and its relevant instances.

    For every set.
    However, for train, determninistic is set to False to allow dropout over the patches IF requiested.
    Over valid an test sets, deterministic is True.

    :param args: object. Contains the configuration of the exp that has been read from the yaml file.
    :param deterministic: True/False. If True, dropping some samples will be allowed IF it was requested. Should set
           to True only with the train set.
    :return: an instance of a patch splitter.
    """
    assert args.patch_splitter is not None, "We need a patch splitter, and you didn't specify one! .... [NOT OK]"
    patch_splitter_conf = Dict2Obj(args.patch_splitter)
    random_cropper = Dict2Obj(args.random_cropper)
    if patch_splitter_conf.name == "PatchSplitter":
        keep = 1.  # default value for deterministic scenario: keep all patch (evaluation phase).
        if not deterministic:
            keep = patch_splitter_conf.keep

        h = patch_splitter_conf.h
        w = patch_splitter_conf.w
        h_ = patch_splitter_conf.h_
        w_ = patch_splitter_conf.w_

        # Instantiate the patch transforms if there is any.
        patch_transform = None
        if patch_splitter_conf.patch_transform:
            error_msg = "We support only one or none patch transform for now ... [NOT OK]"
            assert not isinstance(patch_splitter_conf.patch_transform, Sequence), error_msg

            patch_transform_config = Dict2Obj(patch_splitter_conf.patch_transform)
            if patch_transform_config.name == "PseudoFoveation":
                scale_factor = patch_transform_config.scale_factor
                int_eps = patch_transform_config.int_eps
                num_workers = patch_transform_config.num_workers

                patch_transform = loader.__dict__["PseudoFoveation"](h, w, h_, w_, scale_factor, int_eps, num_workers)

                print(
                    "Patch transform `{}` was successfully instantiated WITHIN a patch splitter `{}`"
                    "with `{}` workers.... [OK]".format(
                        patch_transform_config.name, patch_splitter_conf.name, num_workers)
                )

            elif patch_transform_config.name == "FastApproximationPseudoFoveation":
                scale_factor = patch_transform_config.scale_factor
                int_eps = patch_transform_config.int_eps
                nbr_kernels = patch_transform_config.nbr_kernels
                use_gpu = patch_transform_config.use_gpu
                gpu_id = patch_transform_config.gpu_id

                if gpu_id is None:
                    gpu_id = int(args.cudaid)
                    warnings.warn("You didn't specify the CUDA device ID to run `FastApproximationPseudoFoveation`. "
                                  "We set it up to the same device where the model will be run `cuda:{}` .... [NOT "
                                  "OK]".format(args.cudaid))

                assert args.num_workers in [0, 1], "'config.num_workers' must be in {0, " \
                                                   "1} if loader.FastApproximationPseudoFoveation() is used. " \
                                                   "Multiprocessing does not play well when Dataloader has uses also " \
                                                   "multiprocessing .... [NOT OK]"

                patch_transform = loader.__dict__["FastApproximationPseudoFoveation"](
                    h, w, h_, w_, scale_factor, int_eps, nbr_kernels, use_gpu, gpu_id
                )

                print(
                    "Patch transform `{}` was successfully instantiated WITHIN a patch splitter `{}` "
                    "with `{}` kernels with `{}` GPU and CUDA ID `{}` .... [OK]".format(
                        patch_transform_config.name, patch_splitter_conf.name, nbr_kernels, use_gpu, gpu_id)
                )

            else:
                raise ValueError("Unsupported patch transform `{}`  .... [NOT OK]".format(patch_transform_config.name))
        else:
            print("Proceeding WITHOUT any patch transform  ..... [OK]")

        if patch_transform:
            patch_transform = [patch_transform]

        padding_mode = patch_splitter_conf.padding_mode
        assert hasattr(random_cropper, "make_cropped_perfect_for_split"), "The random cropper `{}` does not have the " \
                                                                          "attribute `make_cropped_perfect_for_split`" \
                                                                          "which we expect .... [NO OK]".format(
                                                                          random_cropper.name)
        if random_cropper.make_cropped_perfect_for_split and not deterministic:
            padding_mode = None
        patch_splitter = loader.__dict__["PatchSplitter"](
            h, w, h_, w_, padding_mode, patch_transforms=patch_transform, keep=keep
        )

        print("Patch splitter `{}` was successfully instantiated .... [OK]".format(patch_splitter_conf.name))

    else:
        raise ValueError("Unsupported patch splitter `{}` .... [NOT OK]".format(patch_splitter_conf.name))

    return patch_splitter


def instantiante_random_cropper(args):
    """
    Instantiate a random cropper. It is used for sampling su-images from an original image in the train set.

    Classes are located in loader.*

    :param args: object. Contains the configuration of the exp that has been read from the yaml file.
    :return: an instance of a random cropper, or None.
    """
    if args.random_cropper:
        r_cropper_config = Dict2Obj(args.random_cropper)
        patch_splitter_config = Dict2Obj(args.patch_splitter)

        if r_cropper_config.name == "RandomCropper":
            min_height = r_cropper_config.min_height
            min_width = r_cropper_config.min_width
            max_height = r_cropper_config.max_height
            max_width = r_cropper_config.max_width
            make_cropped_perfect_for_split = r_cropper_config.make_cropped_perfect_for_split
            h, w, h_, w_ = None, None, None, None
            if make_cropped_perfect_for_split:
                assert patch_splitter_config.name == "PatchSplitter", "We expected the class `PatchSplitter`" \
                                                                      "but found `{}` .... [NOT OK]".format(
                                                                       patch_splitter_config.name)
                h = patch_splitter_config.h
                w = patch_splitter_config.w
                h_ = patch_splitter_config.h_
                w_ = patch_splitter_config.w_

            random_cropper = loader.__dict__["RandomCropper"](
                min_height, min_width, max_height, max_width, make_cropped_perfect_for_split, h, w, h_, w_)

            print("Random cropper `{}` was successfully instantiated .... [OK]".format(r_cropper_config.name))

            return random_cropper

        else:
            raise ValueError("Unsuppoerted random cropper `{}` .... [NOT OK]".format(r_cropper_config.name))
    else:
        return None


