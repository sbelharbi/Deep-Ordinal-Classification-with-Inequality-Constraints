import os
import datetime as dt
import copy
from os.path import join, basename
import subprocess
from shutil import copy as shcopy


import tqdm
import numpy as np
from scipy.special import softmax
import torch
import pickle as pkl


from tools import log, VisualisePP, VisualiseOverlDist
from deeplearning.criteria import Metrics
from reproducibility import reset_seed


def train_one_epoch(model, optimizer, dataloader, criterion, device, tr_stats,
                    epoch=0, log_file=None, ALLOW_MULTIGPUS=False, NBRGPUS=1):
    """
    Perform one epoch of training.
    :param model: instance of a model.
    :param optimizer: instance of an optimizer.
    :param dataloader: instance of a dataloader.
    :param criterion: instance of a learning criterion.
    :param device: a device.
    :param tr_stats: numpy matrix that holds the states of the training. or
    None.
    :param epoch: int, the current epoch.
    :param log_file: a logfile.
    :param ALLOW_MULTIGPUS: bool. If True, we are in multiGPU mode.
    :param NBRGPUS: int, number of GPUs.
    :return:
    """
    model.train()
    metrics = Metrics().to(device)

    length = len(dataloader)
    t0 = dt.datetime.now()
    # acc, mae, soi_y, soi_py, loss
    tracker = np.zeros((length, 5), dtype=np.float32)

    for i, (data, masks, labels) in tqdm.tqdm(
            enumerate(dataloader), ncols=80, total=length):
        seedx = int(os.environ["MYSEED"]) + epoch
        reset_seed(seedx)
        seedx += i

        data = data.to(device)
        labels = labels.to(device)

        model.zero_grad()
        prngs_cuda = None

        # Optimization:
        if not ALLOW_MULTIGPUS:
            if "CC_CLUSTER" in os.environ.keys():
                msg = "Something wrong. You deactivated multigpu mode, " \
                      "but we find {} GPUs. This will not guarantee " \
                      "reproducibility. We do not know why you did that. " \
                      "Exiting .... [NOT OK]".format(NBRGPUS)
                assert NBRGPUS <= 1, msg
            seeds_threads = None
        else:
            msg = "Something is wrong. You asked for multigpu mode. But, " \
                  "we found {} GPUs. Exiting .... [NOT OK]".format(NBRGPUS)
            assert NBRGPUS > 1, msg
            # The seeds are generated randomly before calling the threads.
            reset_seed(seedx)
            seeds_threads = torch.randint(
                0, np.iinfo(np.uint32).max + 1, (NBRGPUS, )).to(device)
            reset_seed(seedx)
            prngs_cuda = []
            # Create different prng states of cuda before forking.
            for seed in seeds_threads:
                # get the corresponding state of the cuda prng with respect to
                # the seed.
                inter_seed = seed.cpu().item()
                # change the internal state of the prng to a random one using
                # the random seed so to capture it.
                torch.manual_seed(inter_seed)
                torch.cuda.manual_seed(inter_seed)
                # capture the prng state.
                prngs_cuda.append(torch.cuda.get_rng_state())
            reset_seed(seedx)

        if prngs_cuda is not None and prngs_cuda != []:
            prngs_cuda = torch.stack(prngs_cuda)

        reset_seed(seedx)
        scores, _ = model(x=data, seed=seeds_threads, prngs_cuda=prngs_cuda)
        reset_seed(seedx)
        loss = criterion(scores, labels)
        reset_seed(seedx)
        loss.backward()
        reset_seed(seedx)
        # Update params.
        optimizer.step()
        reset_seed(seedx)
        # End optimization.
        tracker[i, -1] = loss.item()

        # metrics
        batch_metrics = metrics(
            scores=scores, labels=labels, tr_loss=criterion,
            avg=True).cpu().numpy()
        tracker[i, :-1] = batch_metrics

    t_lb = 0.
    if hasattr(criterion.lossCT, "t_lb"):
        t_lb = criterion.lossCT.t_lb.item()  # assume gpu.

    to_write = "Tr.Ep {:>2d}: ACC: {:.4f}, MAE: {:.4f}, SOI_Y: {:.4f}, " \
               "SOI_PY: {:.4f}, Loss: {:.4f}, LR: {}, t: {:.4f}, " \
               "time:{}".format(
                epoch, tracker[:, 0].mean(), tracker[:, 1].mean(),
                tracker[:, 2].mean(), tracker[:, 3].mean(),
                tracker[:, 4].mean(),
                ['{:.2e}'.format(group["lr"]) for group in
                 optimizer.param_groups], t_lb,
                dt.datetime.now() - t0
                )
    print(to_write)
    if log_file:
        log(log_file, to_write)

    # Update stats:
    if tr_stats is not None:
        tr_stats = np.vstack([tr_stats, tracker])
    else:
        tr_stats = copy.deepcopy(tracker)
    return tr_stats


def validate(model, dataloader, criterion, device, stats, epoch=0,
             log_file=None):
    """
    Perform a validation over the validation set. Assumes a batch size of 1.
    (images do not have the same size, so we can't stack them in one tensor).
    Validation samples may be large to fit all in the GPU at once.
    """
    model.eval()
    metrics = Metrics().to(device)

    length = len(dataloader)
    # acc, mae, soi_y, soi_py, loss
    tracker = np.zeros((1, 5), dtype=np.float32)
    t0 = dt.datetime.now()

    with torch.no_grad():
        for i, (data, mask, label) in tqdm.tqdm(
                enumerate(dataloader), ncols=80, total=length):

            reset_seed(int(os.environ["MYSEED"]) + epoch)

            msg = "Expected a batch size of 1. Found `{}`  .... " \
                  "[NOT OK]".format(data.size()[0])
            assert data.size()[0] == 1, msg

            data = data.to(device)
            labels = label.to(device)

            # In validation, we do not need reproducibility since everything
            # is expected to deterministic. Plus,
            # we use only one gpu since the batch size os 1.
            scores, _ = model(x=data, seed=None)
            loss = criterion(scores, labels)
            batch_metrics = metrics(
                scores=scores, labels=labels, tr_loss=criterion,
                avg=False).cpu().numpy()

            tracker[0, -1] += loss.item()
            tracker[0, :-1] += batch_metrics

    tracker /= float(length)
    t_lb = 0.
    if hasattr(criterion.lossCT, "t_lb"):
        t_lb = criterion.lossCT.t_lb.item()  # assume gpu.

    to_write = "Vl.Ep {:>2d}: ACC: {:.4f}, MAE: {:.4f}, SOI_Y: {:.4f}, " \
               "SOI_PY: {:.4f}, Loss: {:.4f}, t:{:.4f}, time:{}".format(
                epoch, tracker[:, 0].mean(), tracker[:, 1].mean(),
                tracker[:, 2].mean(), tracker[:, 3].mean(),
                tracker[:, 4].mean(), t_lb,
                dt.datetime.now() - t0)
    print(to_write)
    if log_file:
        log(log_file, to_write)

    # Update stats
    if stats is not None:
        stats = np.vstack([stats, tracker])
    else:
        stats = copy.deepcopy(tracker)
    return stats


def final_validate(model, dataloader, criterion, device, dataset, outd,
                   log_file=None, name_set=""):
    """
    Perform a validation over the validation set. Assumes a batch size of 1.
    (images do not have the same size, so we can't stack them in one tensor).
    Validation samples may be large to fit all in the GPU at once.

    :param outd: str, output directory of this dataset.
    :param name_set: str, name to indicate which set is being processed. e.g.:
    trainset, validset, testset.
    """
    visualisor = VisualisePP(floating=4, height_tag=60)
    outd_data = join(outd, "prediction")
    if not os.path.exists(outd_data):
        os.makedirs(outd_data)

    # Deal with overloaded quota of files on servers: use the node disc.
    FOLDER = ""
    if "CC_CLUSTER" in os.environ.keys():
        FOLDER = join(os.environ["SLURM_TMPDIR"], "prediction")
        if not os.path.exists(FOLDER):
            os.makedirs(FOLDER)
    model.eval()
    metrics = Metrics().to(device)

    length = len(dataloader)
    num_classes = len(list(dataset.name_classes.keys()))
    # acc, mae, soi_y, soi_py, loss
    tracker = np.zeros((length, 5 + num_classes), dtype=np.float32)

    t0 = dt.datetime.now()

    with torch.no_grad():
        for i, (data, mask, label) in tqdm.tqdm(
                enumerate(dataloader), ncols=80, total=length):

            reset_seed(int(os.environ["MYSEED"]))

            msg = "Expected a batch size of 1. Found `{}`  .... " \
                  "[NOT OK]".format(data.size()[0])
            assert data.size()[0] == 1, msg

            data = data.to(device)
            labels = label.to(device)

            # In validation, we do not need reproducibility since everything
            # is expected to be deterministic. Plus,
            # we use only one gpu since the batch size os 1.
            scores, _ = model(x=data, seed=None)
            loss = criterion(scores, labels)
            batch_metrics = metrics(
                scores=scores, labels=labels, tr_loss=criterion,
                avg=False).cpu().numpy()

            tracker[i, 4] = loss.item()
            tracker[i, :4] = batch_metrics
            tracker[i, 5:] = softmax(scores.cpu().detach().numpy())

            basef = basename(dataset.get_path_input_img(i))
            img_out = visualisor(
                input_img=dataset.get_original_input_img(i),
                stats=[tracker[i, :]],
                label=dataset.get_original_input_label_int(i),
                name_classes=dataset.name_classes,
                loss_name=[criterion.literal],
                name_file=basef
            )
            fdout = FOLDER if FOLDER else outd_data
            img_out.save(
                join(fdout, "{}.jpeg".format(basef.split('.')[0])), "JPEG")

    # overlap distributions.
    # vis_over_dis = VisualiseOverlDist()
    # vis_over_dis(tracker[:, 5:], dataset.name_classes, outd)

    # compress, then delete files to prevent overloading the disc quota of
    # number of files.
    source = FOLDER if FOLDER else outd_data
    ex = 'zip'
    try:
        cmd_compress = 'zip -rjq {}.zip {}'.format(source, source)
        print("Run: `{}`".format(cmd_compress))
        # os.system(cmd_compress)
        subprocess.run(cmd_compress, shell=True, check=True)
    except subprocess.CalledProcessError:
        cmd_compress = 'tar -zcf {}.tar.gz -C {} .'.format(source, source)
        print("Run: `{}`".format(cmd_compress))
        # os.system(cmd_compress)
        subprocess.run(cmd_compress, shell=True, check=True)
        ex = 'tar.gz'

    cmd_del = 'rm -r {}'.format(outd_data)
    print("Run: `{}`".format(cmd_del))
    os.system(cmd_del)
    if FOLDER:
        shcopy("{}.{}".format(FOLDER, ex), outd)

    tmp = tracker.mean(axis=0)

    t_lb = 0.
    if hasattr(criterion.lossCT, "t_lb"):
        t_lb = criterion.lossCT.t_lb.item()  # assume gpu.

    to_write = "EVAL.FINAL {}: ACC: {:.4f}, MAE: {:.4f}, SOI_Y: {:.4f}, " \
               "SOI_PY: {:.4f}, Loss: {:.4f}, t:{:.4f}, time:{}".format(
                name_set, tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], t_lb,
                dt.datetime.now() - t0)
    to_write = "{} \n{} \n{}".format(10 * "=", to_write, 10 * "=")
    print(to_write)
    if log_file:
        log(log_file, to_write)

    # store the stats in pickle.
    with open(join(outd, 'tracker-{}.pkl'.format(name_set)), 'wb') as fout:
        pkl.dump(tracker, fout, protocol=pkl.HIGHEST_PROTOCOL)

