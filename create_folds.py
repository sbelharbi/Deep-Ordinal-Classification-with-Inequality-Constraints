"""
Splits the following dataset into k-folds:
1. GlaS.
2. Caltech-UCSD-Birds-200-2011
"""

import glob
from os.path import join, relpath, basename, splitext, isfile, expanduser
import os
import traceback
import random
import sys
import math
import csv
import copy
import getpass
import fnmatch

import yaml
import numpy as np
from scipy.io import loadmat
from PIL import Image, ImageChops
import tqdm
import matplotlib.pyplot as plt


from tools import chunk_it, Dict2Obj, announce_msg, check_if_allow_multgpu_mode

import reproducibility


def find_files_pattern(fd_in_, pattern_):
    """
    Find paths to files with pattern within a folder recursively.
    :return:
    """
    assert os.path.exists(fd_in_), "Folder {} does not exist " \
                                   ".... [NOT OK]".format(fd_in_)
    files = []
    for r, d, f in os.walk(fd_in_):
        for file in f:
            if fnmatch.fnmatch(file, pattern_):
                files.append(os.path.join(r, file))

    return files


def show_msg(ms, lg):
    print(ms)
    lg.write(ms + "\n")


def stats_fgnet(args):
    """
    Check some stuff in FG-NET.
    :param args: object, contain the arguments of splitting.
    :return:
    """
    if not os.path.isdir(args.fold_folder):
        os.makedirs(args.fold_folder)

    log = open(join(args.fold_folder, "log-stats.txt"), 'w')
    announce_msg("Going to check FGNET")

    rootpath = args.baseurl
    if args.dataset == "fgnet":
        rootpath = join(rootpath, "images")
    else:
        raise ValueError(
            "Dataset name {} is unknown.".format(str(args.dataset)))

    samples = glob.glob(join(rootpath, "*." + args.img_extension))
    lh, lw = [], []
    for s in tqdm.tqdm(samples, ncols=80, total=len(samples)):
        im = Image.open(s).convert("RGB")
        w, h = im.size
        lh.append(h)
        lw.append(w)
    msg = "min h {}, \t max h {}".format(min(lh), max(lh))
    show_msg(msg, log)
    msg = "min w {}, \t max w {}".format(min(lw), max(lw))
    show_msg(msg, log)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].hist(lh)
    axes[0].set_title('Heights')
    axes[1].hist(lw)
    axes[1].set_title('Widths')
    fig.tight_layout()
    plt.savefig(join(args.fold_folder, "size-stats-fgnet.png"))

    log.close()


def stats_afad(args):
    """
    Check some stuff in AFAD.
    :param args: object, contain the arguments of splitting.
    :return:
    """
    if not os.path.isdir(args.fold_folder):
        os.makedirs(args.fold_folder)

    log = open(join(args.fold_folder, "log-stats.txt"), 'w')
    announce_msg("Going to check {}".format(args.dataset.upper()))

    rootpath = args.baseurl
    if args.dataset == "afad-lite":
        rootpath = join(rootpath, "AFAD-Lite")
    elif args.dataset == "afad-full":
        rootpath = join(rootpath, "AFAD-Full")
    else:
        raise ValueError(
            "Dataset name {} is unknown.".format(str(args.dataset)))

    samples = glob.glob(
        join(rootpath, "*", "*", "*." + args.img_extension))
    lh, lw = [], []
    for s in tqdm.tqdm(samples, ncols=80, total=len(samples)):
        im = Image.open(s).convert("RGB")
        w, h = im.size
        lh.append(h)
        lw.append(w)
    msg = "min h {}, \t max h {}".format(min(lh), max(lh))
    show_msg(msg, log)
    msg = "min w {}, \t max w {}".format(min(lw), max(lw))
    show_msg(msg, log)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].hist(lh)
    axes[0].set_title('Heights')
    axes[1].hist(lw)
    axes[1].set_title('Widths')
    fig.tight_layout()
    plt.savefig(join(args.fold_folder, "size-stats-{}.png".format(
        args.dataset)))

    log.close()


def stats_historical_color_image_decade(args):
    """
    Check some stuff in Historical color image de for decade classification
    dataset.

    :param args: object, contain the arguments of splitting.
    :return:
    """
    if not os.path.isdir(args.fold_folder):
        os.makedirs(args.fold_folder)

    log = open(join(args.fold_folder, "log-stats.txt"), 'w')
    announce_msg("Going to check Historical color image de for "
                 "decade classification dataset")

    rootpath = args.baseurl
    if args.dataset == "historical-color-image-decade":
        rootpath = join(rootpath, "data/imgs/decade_database")
    else:
        raise ValueError(
            "Dataset name {} is unknown.".format(str(args.dataset)))

    samples = glob.glob(
        join(rootpath, "*", "*." + args.img_extension))
    lh, lw = [], []
    for s in tqdm.tqdm(samples, ncols=80, total=len(samples)):
        im = Image.open(s).convert("RGB")
        w, h = im.size
        lh.append(h)
        lw.append(w)
    msg = "min h {}, \t max h {}".format(min(lh), max(lh))
    show_msg(msg, log)
    msg = "min w {}, \t max w {}".format(min(lw), max(lw))
    show_msg(msg, log)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].hist(lh)
    axes[0].set_title('Heights')
    axes[1].hist(lw)
    axes[1].set_title('Widths')
    fig.tight_layout()
    plt.savefig(
        join(args.fold_folder, "size-stats-historical-color-image-decade.png"))

    log.close()


def split_historical_color_image_decade(args):
    """
    Split Historical color image de for decade classification dataset.
    :param args: object, contain the arguments of splitting.
    :return:
    """
    def dump_fold_into_csv(lsamples, outpath):
        """
        Write a list of RELATIVE paths into a csv file.
        Relative paths allow running the code an any device.
        The absolute path within the device will be determined at the running time.

        csv file format: relative path to the image, class (str).

        :param lsamples: list of str of relative paths. and their label.
        :param outpath: str, output file name.
        :return:
        """
        with open(outpath, 'w') as fcsv:
            filewriter = csv.writer(
                fcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for fname, age in lsamples:
                filewriter.writerow([fname, age])

    if not os.path.isdir(args.fold_folder):
        os.makedirs(args.fold_folder)

    log = open(join(args.fold_folder, "log.txt"), 'w')
    announce_msg("Going to split Historical color image de for decade "
                 "classification into {} times".format(args.nbr_splits))

    rootpath = args.baseurl
    if args.dataset == "historical-color-image-decade":
        rootpath = join(rootpath, "data/imgs/decade_database")
    else:
        raise ValueError(
            "Dataset name {} is unknown.".format(str(args.dataset)))

    samples = glob.glob(
        join(rootpath, "*", "*." + args.img_extension))
    samples = sorted([join(*sx.split(os.sep)[-5:]) for sx in samples])

    # Split through decades.
    decades = dict()
    for s in samples:
        decade = s.split(os.sep)[-2]
        if decade in decades.keys():
            decades[decade].append((s, decade))
        else:
            decades[decade] = [(s, decade)]

    assert len(list(decades.keys())) == 5, "Expected 5 decades. found {}" \
                                           ".... [NOT OK]".format(
        len(list(decades.keys()))
    )
    for k in decades.keys():
        assert len(decades[k]) == 265, 'Excpect 265 sample per decade.' \
                                       'Found {} for {} .... [NOT OK]'.format(
            len(decades[k]), k
        )

    splits = []
    for i in range(args.nbr_splits):
        print("Split {}".format(i))
        for k in decades.keys():
            for t in range(1000):
                random.shuffle(decades[k])

        splits.append(copy.deepcopy(decades))

    readme = "Format: relative path to the image, class (decade, str) \n"
    dict_classes_names = dict()
    name_decades = list(decades.keys())
    assert len(name_decades) == 5, 'Expected 5 decades. found {} ....' \
                                   '[NOT OK]'.format(len(name_decades))
    msg = "We expect the names of the decades to be order this way" \
          "['1930s', '1940s', '1950s', '1960s', '1970s']. found {}" \
          ".... [NOT OK]".format(name_decades)
    assert name_decades == ['1930s', '1940s', '1950s', '1960s', '1970s'], msg

    for i in range(len(name_decades)):
        dict_classes_names[str(name_decades[i])] = i

    for k in decades.keys():
        msg = "Decade {} has {} samples".format(k, len(decades[k]))
        show_msg(msg, log)

    # Create the folds.
    def create_folds_of_one_class(lsamps, s_tr, s_vl):
        """
        Create k folds from a list of samples of the same class, each
        fold contains a train, and valid set with a  predefined size.

        Note: Samples are expected to be shuffled beforehand.

        :param lsamps: list of paths to samples of the same class.
        :param s_tr: int, number of samples in the train set.
        :param s_vl: int, number of samples in the valid set.
        :return: list_folds: list of k tuples (tr_set, vl_set): where each
                 element is the list (str paths)
                 of the samples of each set: train, and valid, respectively.
        """
        assert len(
            lsamps) == s_tr + s_vl, "Something wrong with the provided sizes."

        # chunk the data into chunks of size ts (the size of the test set),
        # so we can rotate the test set.
        list_chunks = list(chunk_it(lsamps, s_vl))
        list_folds = []

        for i in range(len(list_chunks)):
            vl_set = list_chunks[i]

            right, left = [], []
            if i < len(list_chunks) - 1:
                right = list_chunks[i + 1:]
            if i > 0:
                left = list_chunks[:i]

            leftoverchunks = right + left

            leftoversamples = []
            for e in leftoverchunks:
                leftoversamples += e

            tr_set = leftoversamples
            list_folds.append((tr_set, vl_set))

        return list_folds

    # create one split..
    def create_one_split(split_i, c_split):
        """
        Create one split of k-folds.

        :param split_i: int, the id of the split.
        :param c_split: dict, contains the current split.
        :param nbr_folds: int, number of folds [the k value in k-folds].
        :return:
        """
        l_folds_per_class = []
        test_samples = []
        nbr_folds = int(1. / args.folding["vl"])
        for key in c_split.keys():
            # get the fixed test set.
            test_samples.extend(c_split[key][:args.test_portion])
            # remove the test set.
            c_split[key] = c_split[key][args.test_portion:]
            # count the number of tr, vl for this current class.
            vl_size = math.ceil(len(c_split[key]) * args.folding["vl"])
            tr_size = len(c_split[key]) - vl_size
            # Create the folds.
            list_folds = create_folds_of_one_class(
                c_split[key], tr_size, vl_size)

            msg = "Expected {} folds. found {} ...[NOT OK]".format(
                nbr_folds, len(list_folds)
            )
            assert len(list_folds) == nbr_folds, msg

            l_folds_per_class.append(list_folds)

        outd = args.fold_folder
        # Re-arrange the folds.
        for i in range(nbr_folds):
            print("\t Fold: {}".format(i))
            out_fold = join(outd, "split_{}/fold_{}".format(split_i, i))
            if not os.path.exists(out_fold):
                os.makedirs(out_fold)

            # dump the test set
            dump_fold_into_csv(
                test_samples,
                join(out_fold, "test_s_{}_f_{}.csv".format(split_i, i)))

            # dump the train set
            train = []
            for el in l_folds_per_class:
                train += el[i][0]  # 0: tr
            # shuffle
            for t in range(1000):
                random.shuffle(train)

            dump_fold_into_csv(
                train,
                join(out_fold, "train_s_{}_f_{}.csv".format(split_i, i)))

            # dump the valid set
            valid = []
            for el in l_folds_per_class:
                valid += el[i][1]  # 1: vl
            dump_fold_into_csv(
                valid,
                join(out_fold, "valid_s_{}_f_{}.csv".format(split_i, i)))

            # dump the seed
            with open(join(out_fold, "seed.txt"), 'w') as fx:
                fx.write("MYSEED: " + os.environ["MYSEED"])
            # dump the coding.
            with open(join(out_fold, "encoding.yaml"), 'w') as f:
                yaml.dump(dict_classes_names, f)

            with open(join(out_fold, "readme.md"), 'w') as fx:
                fx.write(readme)

    outd = args.fold_folder
    if not os.path.isdir(outd):
        os.makedirs(outd)

    with open(join(outd, "readme.md"), 'w') as fx:
        fx.write(readme)
    # dump the coding.
    with open(join(outd, "encoding.yaml"), 'w') as f:
        yaml.dump(dict_classes_names, f)
    # Creates the splits
    print("Starting splitting...")
    for i in range(args.nbr_splits):
        print("Split: {}".format(i))
        create_one_split(i, copy.deepcopy(splits[i]))

    log.close()
    print(
        "All Historical color image de for decade classification splits "
        "(`{}`) ended with success .... [OK]".format(args.nbr_splits))


def split_fgnet(args):
    """
    Split FG-net dataset.
    :param args: object, contain the arguments of splitting.
    """
    def dump_fold_into_csv(lsamples, outpath):
        """
        Write a list of RELATIVE paths into a csv file.
        Relative paths allow running the code an any device.
        The absolute path within the device will be determined at the running time.

        csv file format: relative path to the image, class (str).

        :param lsamples: list of str of relative paths. and their label.
        :param outpath: str, output file name.
        :return:
        """
        with open(outpath, 'w') as fcsv:
            filewriter = csv.writer(
                fcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for fname, age in lsamples:
                filewriter.writerow([fname, age])

    if not os.path.isdir(args.fold_folder):
        os.makedirs(args.fold_folder)

    log = open(join(args.fold_folder, "log.txt"), 'w')
    announce_msg("Going to split FGNET into {} times".format(args.nbr_splits))

    rootpath = args.baseurl
    if args.dataset == "fgnet":
        rootpath = join(rootpath, "images")
    else:
        raise ValueError(
            "Dataset name {} is unknown.".format(str(args.dataset)))

    samples = glob.glob(join(rootpath, "*." + args.img_extension))
    samples = sorted([join(*sx.split(os.sep)[-2:]) for sx in samples])

    # find the subject to perform the split over them to avoid mixing
    # subjects in different sets (train, valid, test).
    subjects = dict()
    ages = []
    for s in samples:
        id, age = basename(s).split('.')[0].split("A")
        if len(age) > 2:
            print(age, age[:-2])
            age = age[:-2]
        if id in subjects.keys():
            subjects[id].append((s, int(age)))
        else:
            subjects[id] = [(s, int(age))]

        ages.append(int(age))

    nbr_sub = len(list(subjects.keys()))
    assert nbr_sub == 82, "Expected 82 subjects. found {}.... [NOT OK]".format(
        nbr_sub)

    for k in subjects.keys():
        print("Subj. {} \t FILE \t\t AGE".format(k))
        for i in subjects[k]:
            print("\t {} \t {}".format(i[0], i[1]))

    nbr_test = int(nbr_sub * args.test_portion)
    nbr_vl = int((nbr_sub - nbr_test) * args.folding["vl"])
    nbr_tr = nbr_sub - (nbr_test + nbr_vl)
    msg = "Total nb. sub. {} \n Train: {} \n Vl: {} \n Test: {}".format(
        nbr_sub, nbr_tr, nbr_vl, nbr_test
    )
    print(msg)
    log.write(msg + "\n")
    items = list(subjects.keys())
    splits = []
    for i in range(args.nbr_splits):
        for t in range(1000):
            random.shuffle(items)
        splits.append(
            (items[:nbr_tr],  # tr
             items[nbr_tr:nbr_tr + nbr_vl],  # vl
             items[nbr_tr + nbr_vl:])  # test
        )

    def get_samples(splitset, sbjs):
        """
        get samples.
        :param splitset:
        :return:
        """
        tmptr = [sbjs[k] for k in splitset]

        sam = []
        for x in tmptr:
            sam.extend(x)

        return copy.deepcopy(sam)

    outd = args.fold_folder
    readme = "Format: relative path to the image, class (age) \n"
    dict_classes_names = dict()
    msg = "Min-age {}, max-age {}".format(min(ages), max(ages))
    print(msg)
    log.write(msg + "\n")

    for i, j in enumerate(range(min(ages), max(ages) + 1)):
        dict_classes_names[str(j)] = i

    for i, sp in enumerate(splits):
        trsam = get_samples(sp[0], subjects)
        vlsam = get_samples(sp[1], subjects)
        tssam = get_samples(sp[2], subjects)

        for _ in range(1000):
            random.shuffle(trsam)
        msg = "Split {}:\t tr {} vl {} tst {}".format(
            i, len(trsam), len(vlsam), len(tssam))
        print(msg)
        log.write(msg + "\n")
        out_fold = join(outd, "split_{}/fold_{}".format(i, 0))
        if not os.path.exists(out_fold):
            os.makedirs(out_fold)

        dump_fold_into_csv(
            trsam, join(out_fold, "train_s_{}_f_0.csv".format(i)))
        dump_fold_into_csv(
            vlsam, join(out_fold, "valid_s_{}_f_0.csv".format(i)))
        dump_fold_into_csv(
            tssam, join(out_fold, "test_s_{}_f_0.csv".format(i)))

        with open(join(out_fold, "seed.txt"), 'w') as fx:
            fx.write("MYSEED: " + os.environ["MYSEED"])
        with open(join(out_fold, "readme.md"), 'w') as fx:
            fx.write(readme)

        with open(join(out_fold, "encoding.yaml"), 'w') as f:
            yaml.dump(dict_classes_names, f)

    log.close()
    with open(join(args.fold_folder, "readme.md"), 'w') as fx:
        fx.write(readme)
    with open(join(args.fold_folder, "encoding.yaml"), 'w') as f:
        yaml.dump(dict_classes_names, f)

    print("All FGNET splits (`{}`) ended with success .... [OK]".format(
        args.nbr_splits))


def split_afad(args):
    """
    Split AFAD-Lite -Full dataset.
    :param args: object, contain the arguments of splitting.
    """
    def dump_fold_into_csv(lsamples, outpath):
        """
        Write a list of RELATIVE paths into a csv file.
        Relative paths allow running the code an any device.
        The absolute path within the device will be determined at the running time.

        csv file format: relative path to the image, class (str).

        :param lsamples: list of str of relative paths. and their label.
        :param outpath: str, output file name.
        :return:
        """
        with open(outpath, 'w') as fcsv:
            filewriter = csv.writer(
                fcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for fname, age in lsamples:
                filewriter.writerow([fname, age])

    if not os.path.isdir(args.fold_folder):
        os.makedirs(args.fold_folder)

    log = open(join(args.fold_folder, "log.txt"), 'w')
    announce_msg("Going to split {} into {} times".format(
        args.dataset,  args.nbr_splits))

    rootpath = args.baseurl
    if args.dataset == "afad-lite":
        rootpath = join(rootpath, "AFAD-Lite")
    elif args.dataset == "afad-full":
        rootpath = join(rootpath, "AFAD-Full")
    else:
        raise ValueError(
            "Dataset name {} is unknown.".format(str(args.dataset)))

    samples = glob.glob(
        join(rootpath, "*", "*", "*." + args.img_extension))
    samples = sorted([join(*sx.split(os.sep)[-4:]) for sx in samples])

    # Split through gender, then, through ages.
    # 111 stands for "male", 112 for "female".
    gendres = {'111': dict(), '112': dict()}
    nbr_male, nbr_female = 0., 0.
    lages = []
    for s in samples:
        age = s.split(os.sep)[-3]
        gendre = s.split(os.sep)[-2]
        msg_x = "{} is unknown gendre. Accepted '111', '112' .... [NOT " \
                "OK]".format(gendre)
        assert gendre in ['111', '112'], msg_x
        if age in gendres[gendre].keys():
            gendres[gendre][age].append((s, int(age)))
        else:
            gendres[gendre][age] = [(s, int(age))]
        lages.append(int(age))
        if gendre == '111':
            nbr_male += 1
        else:
            nbr_female += 1

    msg = "NBR male {}, NBR female {} \n Total: {}".format(
        nbr_male, nbr_female, nbr_male + nbr_female)
    print(msg)
    log.write(msg + "\n")

    if args.dataset == "afad-lite":
        msg = "Expected male samples 34817. Found {} ....[NOT OK]".format(
            nbr_male)
        assert nbr_male == 34817, msg
        msg = "Expected female samples 24527. Found {} ....[NOT OK]".format(
            nbr_female)
        assert nbr_female == 24527, msg
    elif args.dataset == "afad-full":
        msg = "Expected male samples 101519. Found {} ....[NOT OK]".format(
            nbr_male)
        assert nbr_male == 101519, msg
        msg = "Expected female samples 63982. Found {} ....[NOT OK]".format(
            nbr_female)
        assert nbr_female == 63982, msg
    splits = []
    for i in range(args.nbr_splits):
        print("Split {}".format(i))
        sp = ([], [], [])  # tr, vl, tst
        for gn in gendres.keys():
            for age in gendres[gn].keys():
                sm = gendres[gn][age]
                for t in range(100):
                    random.shuffle(sm)

                xs = len(sm)
                nbr_test = int(xs * args.test_portion)
                nbr_vl = int((xs - nbr_test) * args.folding["vl"])
                nbr_tr = xs - (nbr_test + nbr_vl)
                sp[0].extend(sm[:nbr_tr])  # tr
                sp[1].extend(sm[nbr_tr:nbr_tr + nbr_vl])  # vl
                sp[2].extend(sm[nbr_tr + nbr_vl:])  # test

                # start shuffling from here.
                gendres[gn][age] = copy.deepcopy(sm)
        splits.append(copy.deepcopy(sp))

    outd = args.fold_folder
    readme = "Format: relative path to the image, class (age) \n"
    dict_classes_names = dict()
    msg = "Min-age {}, max-age {} \n NBR ages with samples: {}".format(
        min(lages),  max(lages), len(set(lages)))
    print(msg)
    log.write(msg + "\n")

    log.write("{} Encoding {}\n".format(10 * "=", 10 * "="))

    # Classes should be contiguous from the smallest age to the maximum.
    for i, j in enumerate(range(min(lages), max(lages) + 1)):
        dict_classes_names[str(j)] = i
        log.write("`{}`: {} \n".format(j, i))

    log.write("{} End encoding {}\n".format(10 * "=", 10 * "="))

    for i, sp in enumerate(splits):
        trsam, vlsam, tssam = sp

        for _ in range(1000):
            random.shuffle(trsam)
        msg = "Split {}:\t tr {} vl {} tst {}".format(
            i, len(trsam), len(vlsam), len(tssam))
        print(msg)
        log.write(msg + "\n")
        out_fold = join(outd, "split_{}/fold_{}".format(i, 0))
        if not os.path.exists(out_fold):
            os.makedirs(out_fold)

        dump_fold_into_csv(
            trsam, join(out_fold, "train_s_{}_f_0.csv".format(i)))
        dump_fold_into_csv(
            vlsam, join(out_fold, "valid_s_{}_f_0.csv".format(i)))
        dump_fold_into_csv(
            tssam, join(out_fold, "test_s_{}_f_0.csv".format(i)))

        with open(join(out_fold, "seed.txt"), 'w') as fx:
            fx.write("MYSEED: " + os.environ["MYSEED"])
        with open(join(out_fold, "readme.md"), 'w') as fx:
            fx.write(readme)

        with open(join(out_fold, "encoding.yaml"), 'w') as f:
            yaml.dump(dict_classes_names, f)

    log.close()
    with open(join(args.fold_folder, "readme.md"), 'w') as fx:
        fx.write(readme)
    with open(join(args.fold_folder, "encoding.yaml"), 'w') as f:
        yaml.dump(dict_classes_names, f)

    print("All {} splits (`{}`) ended with success .... [OK]".format(
        args.dataset.upper(), args.nbr_splits))


def create_k_folds_csv_bach_part_a(args):
    """
    Create k folds of the dataset BACH (part A) 2018 and store the image path
    of each fold in a *.csv file.

    1. Test set if fixed for all the splits/folds.
    2. We do a k-fold over the remaining data to create train, and validation
    sets.

    :param args: object, contain the arguments of splitting.
    :return:
    """
    announce_msg("Going to create the  splits, "
                 "and the k-folds fro BACH (PART A) 2018 .... [OK]")

    rootpath = args.baseurl
    if args.dataset == "bach-part-a-2018":
        rootpath = join(rootpath, "ICIAR2018_BACH_Challenge/Photos")
    else:
        raise ValueError(
            "Dataset name {} is unknown.".format(str(args.dataset)))

    samples = glob.glob(join(rootpath, "*", "*." + args.img_extension))
    # Originally, the function was written where 'samples' contains the
    # absolute paths to the files. Then, we realise that using absolute path
    # on different platforms leads to a non-deterministic folds even with the
    # seed fixed. This is a result of glob.glob() that returns a list of paths
    # more likely depending on the how the files are saved within the OS.
    # Therefore, to get rid of this, we use only a short path that is constant
    # across all the hosts where our dataset is saved. Then, we sort the list
    # of paths BEFORE we go further. This will guarantee that whatever the OS
    # the code is running in, the sorted list is the same.
    samples = sorted([join(*sx.split(os.sep)[-4:]) for sx in samples])

    classes = {key: [s for s in samples if s.split(os.sep)[-2] == key] for key in args.name_classes.keys()}

    all_train = {}
    test_fix = []
    # Shuffle to avoid any bias.
    for key in classes.keys():
        for i in range(1000):
            random.shuffle(classes[key])

        nbr_test = int(len(classes[key]) * args.test_portion)
        test_fix += classes[key][:nbr_test]
        all_train[key] = classes[key][nbr_test:]

    # Test set is ready. Now, we need to do k-fold over the train.

    # Create the splits over the train
    splits = []
    for i in range(args.nbr_splits):
        for t in range(1000):
            for k in all_train.keys():
                random.shuffle(all_train[k])

        splits.append(copy.deepcopy(all_train))

    readme = "csv format:\n" \
             "relative path to the image file, class (string)\n" \
             "Example:\n" \
             "ICIAR2018_BACH_Challenge/Photos/Normal/n047.tif, 'Normal'\n" \
             "There are four classes: normal, benign, in situ, and invasive.\n" \
             "The class of the sample may be infered from the parent folder of the image (in the example " \
             "above:\n " \
             "Normal:\n" \
             "Normal: class 'normal'\n" \
             "Benign: class 'benign'\n" \
             "InSitu: class 'in situ'\n" \
             "Invasive: class 'invasive'"

    # Create k-folds for each split.
    def create_folds_of_one_class(lsamps, s_tr, s_vl):
        """
        Create k folds from a list of samples of the same class, each fold contains a train, and valid set with a
        predefined size.

        Samples need to be shuffled beforehand.

        :param lsamps: list of paths to samples of the same class.
        :param s_tr: int, number of samples in the train set.
        :param s_vl: int, number of samples in the valid set.
        :return: list_folds: list of k tuples (tr_set, vl_set, ts_set): where each element is the list (str paths)
                 of the samples of each set: train, valid, and test, respectively.
        """
        assert len(lsamps) == s_tr + s_vl, "Something wrong with the provided sizes .... [NOT OK]"

        # chunk the data into chunks of size ts (the size of the test set), so we can rotate the test set.
        list_chunks = list(chunk_it(lsamps, s_vl))
        list_folds = []

        for i in range(len(list_chunks)):
            vl_set = list_chunks[i]

            right, left = [], []
            if i < len(list_chunks) - 1:
                right = list_chunks[i + 1:]
            if i > 0:
                left = list_chunks[:i]

            leftoverchunks = right + left

            leftoversamples = []
            for e in leftoverchunks:
                leftoversamples += e

            tr_set = leftoversamples
            list_folds.append((tr_set, vl_set))

        return list_folds

    # Save the folds into *.csv files.
    def dump_fold_into_csv(lsamples, outpath):
        """
        Write a list of RELATIVE paths into a csv file.
        Relative paths allow running the code an any device.
        The absolute path within the device will be determined at the running time.

        csv file format: relative path to the image, class (str).

        :param lsamples: list of str of relative paths. and their label.
        :param outpath: str, output file name.
        :return:
        """
        with open(outpath, 'w') as fcsv:
            filewriter = csv.writer(
                fcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for fname in lsamples:
                label = fname.split(os.sep)[-2]
                filewriter.writerow([fname, label])

    def create_one_split(split_i, test_samples, train_samples_all, nbr_folds):
        """
        Create one split of k-folds.

        :param split_i: int, the id of the split.
        :param test_samples: dict of list, each key represents a class (test set, fixed).
        :param train_samples_all: dict of list, each key represent a class (all train set).
        :param nbr_folds: int, number of folds [the k value in k-folds].
        :return:
        """
        # Create the k-folds
        list_folds_of_class = {}

        for key in train_samples_all.keys():
            vl_size = math.ceil(len(train_samples_all[key]) * args.folding["vl"] / 100.)
            tr_size = len(train_samples_all[key]) - vl_size
            list_folds_of_class[key] = create_folds_of_one_class(train_samples_all[key], tr_size, vl_size)

            assert len(list_folds_of_class[key]) == nbr_folds, "We didn't get `{}` folds, but `{}` .... " \
                                                               "[NOT OK]".format(
                nbr_folds, len(list_folds_of_class[key]))

            print("We obtained `{}` folds for the class {}.... [OK]".format(args.nbr_folds, key))

        outd = args.fold_folder
        for i in range(nbr_folds):
            print("Fold {}:\n\t".format(i))
            out_fold = join(outd, "split_" + str(split_i) + "/fold_" + str(i))
            if not os.path.exists(out_fold):
                os.makedirs(out_fold)

            # dump the test set
            dump_fold_into_csv(test_samples, join(out_fold, "test_s_" + str(split_i) + "_f_" + str(i) + ".csv"))

            train = []
            valid = []
            for key in list_folds_of_class.keys():
                # Train
                train += list_folds_of_class[key][i][0]

                # Valid.
                valid += list_folds_of_class[key][i][1]

            # shuffle
            for t in range(1000):
                random.shuffle(train)

            dump_fold_into_csv(train, join(out_fold, "train_s_" + str(split_i) + "_f_" + str(i) + ".csv"))
            dump_fold_into_csv(valid, join(out_fold, "valid_s_" + str(split_i) + "_f_" + str(i) + ".csv"))

            # dump the seed
            with open(join(out_fold, "seed.txt"), 'w') as fx:
                fx.write("MYSEED: " + os.environ["MYSEED"])

            with open(join(out_fold, "readme.md"), 'w') as fx:
                fx.write(readme)
        print("BACH (PART A) 2018 splitting N° `{}` ends with success .... [OK]".format(split_i))

    if not os.path.isdir(args.fold_folder):
        os.makedirs(args.fold_folder)

    # Creates the splits
    for i in range(args.nbr_splits):
        print("Split {}:\n\t".format(i))
        create_one_split(i, test_fix, splits[i], args.nbr_folds)

    with open(join(args.fold_folder, "readme.md"), 'w') as fx:
        fx.write(readme)
    print("All BACH (PART A) 2018 splitting (`{}`) ended with success .... [OK]".format(args.nbr_splits))


def split_valid_glas(args):
    """
    Create a validation/train sets in GlaS dataset.
    csv file format: relative path to the image, relative path to the mask, class (str: benign, malignant).

    :param args:
    :return:
    """
    classes = ["benign", "malignant"]
    all_samples = []
    # Read the file Grade.csv
    baseurl = args.baseurl
    with open(join(baseurl, "Grade.csv"), 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # get rid of the header.
        for row in reader:
            # not sure why they thought it is a good idea to put a space before the class. Now, I have to get rid of
            # it and possibly other hidden spaces ...
            row = [r.replace(" ", "") for r in row]
            assert row[2] in classes, "The class `{}` is not within the predefined classes `{}`".format(row[2], classes)
            all_samples.append([row[0], row[2]])

    assert len(all_samples) == 165, "The number of samples {} do not match what they said (165) .... [NOT " \
                                    "OK]".format(len(all_samples))

    # Take test samples aside. They are fix.
    test_samples = [s for s in all_samples if s[0].startswith("test")]
    assert len(test_samples) == 80, "The number of test samples {} is not 80 as they said .... [NOT OK]".format(len(
        test_samples))

    all_train_samples = [s for s in all_samples if s[0].startswith("train")]
    assert len(all_train_samples) == 85, "The number of train samples {} is not 85 as they said .... [NOT OK]".format(
        len(all_train_samples))

    benign = [s for s in all_train_samples if s[1] == "benign"]
    malignant = [s for s in all_train_samples if s[1] == "malignant"]

    # Split
    splits = []
    for i in range(args.nbr_splits):
        for _ in range(1000):
            random.shuffle(benign)
            random.shuffle(malignant)
        splits.append({"benign": copy.deepcopy(benign),
                       "malignant": copy.deepcopy(malignant)}
                      )

    # Create the folds.
    def create_folds_of_one_class(lsamps, s_tr, s_vl):
        """
        Create k folds from a list of samples of the same class, each fold contains a train, and valid set with a
        predefined size.

        Note: Samples are expected to be shuffled beforehand.

        :param lsamps: list of paths to samples of the same class.
        :param s_tr: int, number of samples in the train set.
        :param s_vl: int, number of samples in the valid set.
        :return: list_folds: list of k tuples (tr_set, vl_set, ts_set): where each element is the list (str paths)
                 of the samples of each set: train, valid, and test, respectively.
        """
        assert len(lsamps) == s_tr + s_vl, "Something wrong with the provided sizes."

        # chunk the data into chunks of size ts (the size of the test set), so we can rotate the test set.
        list_chunks = list(chunk_it(lsamps, s_vl))
        list_folds = []

        for i in range(len(list_chunks)):
            vl_set = list_chunks[i]

            right, left = [], []
            if i < len(list_chunks) - 1:
                right = list_chunks[i + 1:]
            if i > 0:
                left = list_chunks[:i]

            leftoverchunks = right + left

            leftoversamples = []
            for e in leftoverchunks:
                leftoversamples += e

            tr_set = leftoversamples
            list_folds.append((tr_set, vl_set))

        return list_folds

    # Save the folds.
    # Save the folds into *.csv files.
    def dump_fold_into_csv(lsamples, outpath):
        """
        Write a list of RELATIVE paths into a csv file.
        Relative paths allow running the code an any device.
        The absolute path within the device will be determined at the running time.

        csv file format: relative path to the image, relative path to the mask, class (str: benign, malignant).

        :param lsamples: list of str of relative paths.
        :param outpath: str, output file name.
        :return:
        """
        with open(outpath, 'w') as fcsv:
            filewriter = csv.writer(fcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for name, clas in lsamples:
                filewriter.writerow([name + ".bmp", name + "_anno.bmp", clas])

    def create_one_split(split_i, test_samples, benign, malignant, nbr_folds):
        """
        Create one split of k-folds.

        :param split_i: int, the id of the split.
        :param test_samples: list, list of test samples.
        :param benign: list, list of train benign samples.
        :param malignant: list, list of train maligant samples.
        :param nbr_folds: int, number of folds [the k value in k-folds].
        :return:
        """
        vl_size_benign = math.ceil(len(benign) * args.folding["vl"] / 100.)
        vl_size_malignant = math.ceil(len(malignant) * args.folding["vl"] / 100.)

        list_folds_benign = create_folds_of_one_class(benign, len(benign) - vl_size_benign, vl_size_benign)
        list_folds_malignant = create_folds_of_one_class(malignant, len(malignant) - vl_size_malignant,
                                                         vl_size_malignant)

        assert len(list_folds_benign) == len(list_folds_malignant), "We didn't obtain the same number of fold" \
                                                                    " .... [NOT OK]"
        assert len(list_folds_benign) == 5, "We did not get exactly 5 folds, but `{}` .... [ NOT OK]".format(
            len(list_folds_benign))
        print("We found {} folds .... [OK]".format(len(list_folds_malignant)))

        outd = args.fold_folder
        for i in range(nbr_folds):
            out_fold = join(outd, "split_" + str(split_i) + "/fold_" + str(i))
            if not os.path.exists(out_fold):
                os.makedirs(out_fold)

            # dump the test set
            dump_fold_into_csv(test_samples, join(out_fold, "test_s_" + str(split_i) + "_f_" + str(i) + ".csv"))

            # dump the train set
            train = list_folds_malignant[i][0] + list_folds_benign[i][0]
            # shuffle
            for t in range(1000):
                random.shuffle(train)

            dump_fold_into_csv(train, join(out_fold, "train_s_" + str(split_i) + "_f_" + str(i) + ".csv"))

            # dump the valid set
            valid = list_folds_malignant[i][1] + list_folds_benign[i][1]
            dump_fold_into_csv(valid, join(out_fold, "valid_s_" + str(split_i) + "_f_" + str(i) + ".csv"))

            # dump the seed
            with open(join(out_fold, "seed.txt"), 'w') as fx:
                fx.write("MYSEED: " + os.environ["MYSEED"])

        with open(join(outd, "readme.md"), 'w') as fx:
            fx.write("csv format:\nrelative path to the image, relative path to the mask, class "
                     "(str: benign, malignant).")

    if not os.path.isdir(args.fold_folder):
        os.makedirs(args.fold_folder)

    with open(join(args.fold_folder, "readme.md"), 'w') as fx:
        fx.write("csv format:\nrelative path to the image, relative path to the mask, class "
                 "(str: benign, malignant).")
    # Creates the splits
    for i in range(args.nbr_splits):
        create_one_split(i, test_samples, splits[i]["benign"], splits[i]["malignant"], args.nbr_folds)

    print("All GlaS splitting (`{}`) ended with success .... [OK]".format(args.nbr_splits))


def split_valid_Caltech_UCSD_Birds_200_2011(args):
    """
    Create a validation/train sets in Caltech_UCSD_Birds_200_2011 dataset.
    csv file format: relative path to the image, relative path to the mask, class (str).

    :param args:
    :return:
    """
    baseurl = args.baseurl
    classes_names, classes_id = [], []
    # Load the classes: id class
    with open(join(baseurl, "CUB_200_2011", "classes.txt"), "r") as fcl:
        content = fcl.readlines()
        for el in content:
            el = el.rstrip("\n\r")
            idcl, cl = el.split(" ")
            classes_id.append(idcl)
            classes_names.append(cl)
    # Load the images and their id.
    images_path, images_id = [], []
    with open(join(baseurl, "CUB_200_2011", "images.txt"), "r") as fim:
        content = fim.readlines()
        for el in content:
            el = el.strip("\n\r")
            idim, imgpath = el.split(" ")
            images_id.append(idim)
            images_path.append(imgpath)

    # Load the image labels.
    images_label = (np.zeros(len(images_path)) - 1).tolist()
    with open(join(baseurl, "CUB_200_2011", "image_class_labels.txt"), "r") as flb:
        content = flb.readlines()
        for el in content:
            el = el.strip("\n\r")
            idim, clid = el.split(" ")
            # find the image index correspd. to the image id
            images_label[images_id.index(idim)] = classes_names[classes_id.index(clid)]

    # All what we need is in images_label, images_path. classes_names will be used later to convert class name into
    # integers.
    assert len(images_id) == 11788, "We expect Caltech_UCSD_Birds_200_2011 dataset to have 11788 samples. We found {}" \
                                    ".... [NOT OK]".format(len(images_id))
    all_samples = list(zip(images_path, images_label))  # Not used.

    # Split into train and test.
    all_train_samples = []
    test_samples = []
    with open(join(baseurl, "CUB_200_2011", "train_test_split.txt"), "r") as flb:
        content = flb.readlines()
        for el in content:
            el = el.strip("\n\r")
            idim, st = el.split(" ")
            img_idx = images_id.index(idim)
            img_path = images_path[img_idx]
            img_label = images_label[img_idx]
            filename, file_ext = os.path.splitext(img_path)
            mask_path = join("segmentations", filename + ".png")
            img_path = join("CUB_200_2011", "images", img_path)
            assert os.path.isfile(join(args.baseurl, img_path)), "Image {} does not exist!".format(
                join(args.baseurl, img_path))
            assert os.path.isfile(join(args.baseurl, mask_path)), "Mask {} does not exist!".format(
                join(args.baseurl, mask_path))
            pair = (img_path, mask_path, img_label)
            if st == "1":  # train
                all_train_samples.append(pair)
            elif st == "0":  # test
                test_samples.append(pair)
            else:
                raise ValueError("Expected 0 or 1. Found {} .... [NOT OK]".format(st))

    print("Nbr. ALL train samples: {}".format(len(all_train_samples)))
    print("Nbr. test samples: {}".format(len(test_samples)))

    assert len(all_train_samples) + len(test_samples) == 11788, "Something is wrong. We expected 11788. Found: {}" \
                                                                ".... [NOT OK]".format(
        len(all_train_samples) + len(test_samples))

    # Keep only the required classes:
    if args.nbr_classes is not None:
        fyaml = open(args.path_encoding, 'r')
        contyaml = yaml.load(fyaml)
        keys_l = list(contyaml.keys())
        indexer = np.array(list(range(len(keys_l)))).squeeze()
        select_idx = np.random.choice(indexer, args.nbr_classes, replace=False)
        selected_keys = []
        for idx in select_idx:
            selected_keys.append(keys_l[idx])

        # Drop samples outside the selected classes.
        tmp_all_train = []
        for el in all_train_samples:
            if el[2] in selected_keys:
                tmp_all_train.append(el)
        all_train_samples = tmp_all_train

        tmp_test = []
        for el in test_samples:
            if el[2] in selected_keys:
                tmp_test.append(el)

        test_samples = tmp_test

        classes_names = selected_keys

    # Train: Create dict where a key is the class name, and the value is all the samples that have the same class.

    samples_per_class = dict()
    for cl in classes_names:
        samples_per_class[cl] = [el for el in all_train_samples if el[2] == cl]

    # Split
    splits = []
    print("Shuffling to create splits. May take some time...")
    for i in range(args.nbr_splits):
        for key in samples_per_class.keys():
            for _ in range(1000):
                random.shuffle(samples_per_class[key])
                random.shuffle(samples_per_class[key])
        splits.append(copy.deepcopy(samples_per_class))

    # encode class name into int.
    dict_classes_names = dict()
    for i in range(len(classes_names)):
        dict_classes_names[classes_names[i]] = i

    readme = "csv format:\nrelative path to the image, relative path to the mask, class " \
             "(str). \n You can use the providing encoding of the classes in encoding.yaml"

    # Create the folds.
    def create_folds_of_one_class(lsamps, s_tr, s_vl):
        """
        Create k folds from a list of samples of the same class, each fold
         contains a train, and valid set with a     predefined size.

        Note: Samples are expected to be shuffled beforehand.

        :param lsamps: list of paths to samples of the same class.
        :param s_tr: int, number of samples in the train set.
        :param s_vl: int, number of samples in the valid set.
        :return: list_folds: list of k tuples (tr_set, vl_set): where each
                 element is the list (str paths)
                 of the samples of each set: train, and valid, respectively.
        """
        assert len(lsamps) == s_tr + s_vl, "Something wrong with the" \
                                           " provided sizes."

        # chunk the data into chunks of size ts (the size of the test set),
        # so we can rotate the test set.
        list_chunks = list(chunk_it(lsamps, s_vl))
        list_folds = []

        for i in range(len(list_chunks)):
            vl_set = list_chunks[i]

            right, left = [], []
            if i < len(list_chunks) - 1:
                right = list_chunks[i + 1:]
            if i > 0:
                left = list_chunks[:i]

            leftoverchunks = right + left

            leftoversamples = []
            for e in leftoverchunks:
                leftoversamples += e

            tr_set = leftoversamples
            list_folds.append((tr_set, vl_set))

        return list_folds

    # Save the folds.
    # Save the folds into *.csv files.
    def dump_fold_into_csv(lsamples, outpath):
        """
        Write a list of RELATIVE paths into a csv file.
        Relative paths allow running the code on any device.
        The absolute path within the device will be determined at the running time.

        csv file format: relative path to the image, relative path to the mask, class (str).

        :param lsamples: list of str of relative paths.
        :param outpath: str, output file name.
        :return:
        """
        with open(outpath, 'w') as fcsv:
            filewriter = csv.writer(fcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for im_path, mk_path, cl in lsamples:
                filewriter.writerow([im_path, mk_path, cl])

    def create_one_split(split_i, test_samples, c_split, nbr_folds):
        """
        Create one split of k-folds.

        :param split_i: int, the id of the split.
        :param test_samples: list, list of test samples.
        :param c_split: dict, contains the current split.
        :param nbr_folds: int, number of folds [the k value in k-folds].
        :return:
        """
        l_folds_per_class = []
        for key in c_split.keys():
            # count the number of tr, vl for this current class.
            vl_size = math.ceil(len(c_split[key]) * args.folding["vl"] / 100.)
            tr_size = len(c_split[key]) - vl_size
            # Create the folds.
            list_folds = create_folds_of_one_class(c_split[key], tr_size, vl_size)

            assert len(list_folds) == nbr_folds, "We did not get exactly {} folds, but `{}` .... [ NOT OK]".format(
                nbr_folds,  len(list_folds))

            l_folds_per_class.append(list_folds)

        outd = args.fold_folder
        # Re-arrange the folds.
        for i in range(nbr_folds):
            print("\t Fold: {}".format(i))
            out_fold = join(outd, "split_" + str(split_i) + "/fold_" + str(i))
            if not os.path.exists(out_fold):
                os.makedirs(out_fold)

            # dump the test set
            dump_fold_into_csv(test_samples, join(out_fold, "test_s_" + str(split_i) + "_f_" + str(i) + ".csv"))

            # dump the train set
            train = []
            for el in l_folds_per_class:
                train += el[i][0]  # 0: tr
            # shuffle
            for t in range(1000):
                random.shuffle(train)

            dump_fold_into_csv(train, join(out_fold, "train_s_" + str(split_i) + "_f_" + str(i) + ".csv"))

            # dump the valid set
            valid = []
            for el in l_folds_per_class:
                valid += el[i][1]  # 1: vl
            dump_fold_into_csv(valid, join(out_fold, "valid_s_" + str(split_i) + "_f_" + str(i) + ".csv"))

            # dump the seed
            with open(join(out_fold, "seed.txt"), 'w') as fx:
                fx.write("MYSEED: " + os.environ["MYSEED"])
            # dump the coding.
            with open(join(out_fold, "encoding.yaml"), 'w') as f:
                yaml.dump(dict_classes_names, f)

            with open(join(out_fold, "readme.md"), 'w') as fx:
                fx.write(readme)

    if not os.path.isdir(args.fold_folder):
        os.makedirs(args.fold_folder)

    with open(join(args.fold_folder, "readme.md"), 'w') as fx:
        fx.write(readme)
    # dump the coding.
    with open(join(args.fold_folder, "encoding.yaml"), 'w') as f:
        yaml.dump(dict_classes_names, f)
    # Creates the splits
    print("Starting splitting...")
    for i in range(args.nbr_splits):
        print("Split: {}".format(i))
        create_one_split(i, test_samples, splits[i], args.nbr_folds)

    print(
        "All Caltech_UCSD_Birds_200_2011 splitting (`{}`) ended with "
        "success .... [OK]".format(args.nbr_splits))


def find_files_pattern(fd_in_, pattern_):
    """
    Find paths to files with pattern within a folder recursively.
    :return:
    """
    assert os.path.exists(fd_in_), "Folder {} does not exist .... [NOT OK]".format(fd_in_)
    files = []
    for r, d, f in os.walk(fd_in_):
        for file in f:
            if fnmatch.fnmatch(file, pattern_):
                files.append(os.path.join(r, file))

    return files


def create_bin_mask_Oxford_flowers_102(args):
    """
    Create binary masks.
    :param args:
    :return:
    """
    def get_id(pathx, basex):
        """
        Get the id of a sample.
        :param pathx:
        :return:
        """
        rpath = relpath(pathx, basex)
        basen = basename(rpath)
        id = splitext(basen)[0].split('_')[1]
        return id

    baseurl = args.baseurl
    imgs = find_files_pattern(join(baseurl, 'jpg'), '*.jpg')
    bin_fd = join(baseurl, 'segmim_bin')
    if not os.path.exists(bin_fd):
        os.makedirs(bin_fd)
    else:  # End.
        print('Conversion to binary mask has already been done. [OK]')
        return 0

    # Background color [  0   0 254]. (blue)
    print('Start converting the provided masks into binary masks ....')
    for im in tqdm.tqdm(imgs, ncols=80, total=len(imgs)):
        id_im = get_id(im, baseurl)
        mask = join(baseurl, 'segmim', 'segmim_{}.jpg'.format(id_im))
        assert isfile(mask), 'File {} does not exist. Inconsistent logic. .... [NOT OK]'.format(mask)
        msk_in = Image.open(mask, 'r').convert('RGB')
        arr_ = np.array(msk_in)
        arr_[:, :, 0] = 0
        arr_[:, :, 1] = 0
        arr_[:, :, 2] = 254
        blue = Image.fromarray(arr_.astype(np.uint8), mode='RGB')
        dif = ImageChops.subtract(msk_in, blue)
        x_arr = np.array(dif)
        x_arr = np.mean(x_arr, axis=2)
        x_arr = (x_arr != 0).astype(np.uint8)
        img_bin = Image.fromarray(x_arr * 255, mode='L')
        img_bin.save(join(bin_fd, 'segmim_{}.jpg'.format(id_im)), 'JPEG')


def split_Oxford_flowers_102(args):
    """
    Use the provided split: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat

    :param args:
    :return:
    """
    def dump_fold_into_csv(lsamples, outpath):
        """
        Write a list of RELATIVE paths into a csv file.
        Relative paths allow running the code an any device.
        The absolute path within the device will be determined at the running time.

        csv file format: relative path to the image, relative path to the mask, class (str: int).

        :param lsamples: list of str of relative paths.
        :param outpath: str, output file name.
        :return:
        """
        with open(outpath, 'w') as fcsv:
            filewriter = csv.writer(fcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for imgx, mkx, clx in lsamples:
                filewriter.writerow([imgx, mkx, clx])
    baseurl = args.baseurl

    # splits
    fin = loadmat(join(baseurl, 'setid.mat'))
    trnid = fin['trnid'].reshape((-1)).astype(np.uint16)
    valid = fin['valid'].reshape((-1)).astype(np.uint16)
    tstid = fin['tstid'].reshape((-1)).astype(np.uint16)

    # labels
    flabels = loadmat(join(baseurl, 'imagelabels.mat'))['labels'].flatten()
    flabels -= 1  # labels are encoded from 1 to 102. We change that to be from 0 to 101.

    # find all the files
    fdimg = join(baseurl, 'jpg')
    tr_set, vl_set, ts_set = [], [], []  # (img, mask, label (int))
    filesin = find_files_pattern(fdimg, '*.jpg')
    lid = []
    for f in filesin:
        rpath = relpath(f, baseurl)
        basen = basename(rpath)
        id = splitext(basen)[0].split('_')[1]
        mask = join(baseurl, 'segmim_bin', 'segmim_{}.jpg'.format(id))
        assert isfile(mask), 'File {} does not exist. Inconsistent logic. .... [NOT OK]'.format(mask)
        rpath_mask = relpath(mask, baseurl)
        id = int(id)  # ids start from 1. Array indexing starts from 0.
        label = int(flabels[id - 1])
        sample = (rpath, rpath_mask, label)
        lid.append(id)
        if id in trnid:
            tr_set.append(sample)
        elif id in valid:
            vl_set.append(sample)
        elif id in tstid:
            ts_set.append(sample)
        else:
            raise ValueError('ID:{} not found in train, valid, test. Inconsistent logic. ....[NOT OK]'.format(id))

    print('Number of samples:\n'
          'Train: {} \n'
          'valid: {} \n'
          'Test: {}\n'
          'Toal: {}'.format(len(tr_set), len(vl_set), len(ts_set), len(tr_set) + len(vl_set) + len(ts_set)))

    dict_classes_names = dict()
    uniquel = np.unique(flabels)
    for i in range(uniquel.size):
        dict_classes_names[str(uniquel[i])] = int(uniquel[i])

    outd = args.fold_folder
    out_fold = join(outd, "split_" + str(0) + "/fold_" + str(0))
    if not os.path.exists(out_fold):
        os.makedirs(out_fold)

    dump_fold_into_csv(tr_set, join(out_fold, "train_s_" + str(0) + "_f_" + str(0) + ".csv"))
    dump_fold_into_csv(vl_set, join(out_fold, "valid_s_" + str(0) + "_f_" + str(0) + ".csv"))
    dump_fold_into_csv(ts_set, join(out_fold, "test_s_" + str(0) + "_f_" + str(0) + ".csv"))

    with open(join(out_fold, "encoding.yaml"), 'w') as f:
        yaml.dump(dict_classes_names, f)
    with open(join(args.fold_folder, "encoding.yaml"), 'w') as f:
        yaml.dump(dict_classes_names, f)


# ==============================================================================
#                               RUN
# ==============================================================================
def do_historical_color_image_decade():
    """
    Historical color image datasets for classfication by decade.
    :return:
    """
    # ===============
    # Reproducibility
    # ===============

    # ===========================

    reproducibility.init_seed()

    # ===========================
    # TODO: anonymize.
    if "HOST_XXX" in os.environ.keys():
        if os.environ['HOST_XXX'] == 'laptop':
            baseurl = "{}}/datasets/HistoricalColor-ECCV2012".format(
                os.environ["EXDRIVE"])
        elif os.environ['HOST_XXX'] == 'lab':
            baseurl = "{}/datasets/HistoricalColor-ECCV2012".format(
                os.environ["NEWHOME"])
    elif "CC_CLUSTER" in os.environ.keys():
        baseurl = expanduser(
            "{}/datasets/HistoricalColor-ECCV2012".format(
                os.environ["SCRATCH"]))
    else:
        raise ValueError("Host name unknown. "
                         "Please fix this depending on where you store "
                         "your data .... [NOT OK]")

    args = {"baseurl": baseurl,
            "test_portion": 50,  # number of samples to take from each decade
            # class to for the testset.
            # The left over if for train; and it will
            # be divided into actual train, and validation sets.
            # vl % of train set will be used for validation,
            # while the leftover.
            "folding": {"vl": 0.1},
            # (100-vl)/100% will be used for actual training.
            "dataset": "historical-color-image-decade",
            "fold_folder": "folds/historical-color-image-decade",
            "img_extension": "jpg",
            # how many times to perform the k-fold to create train, valid,
            # and test sets. The test set changes from a split to another.
            "nbr_splits": 10
            }
    split_historical_color_image_decade(Dict2Obj(args))
    stats_historical_color_image_decade(Dict2Obj(args))


def do_afad_full():
    """
    AFAD-Full.
    :return:
    """
    # ===============
    # Reproducibility
    # ===============

    # ===========================

    reproducibility.init_seed()

    # ===========================

    # TODO: anonymize.
    if "HOST_XXX" in os.environ.keys():
        if os.environ['HOST_XXX'] == 'laptop':
            baseurl = "{}/datasets/tarball".format(os.environ["EXDRIVE"])
        elif os.environ['HOST_XXX'] == 'lab':
            baseurl = "{}/datasets/tarball".format(os.environ["NEWHOME"])
    elif "CC_CLUSTER" in os.environ.keys():
        baseurl = expanduser("~/workspace-sc/datasets/tarball")
    else:
        raise ValueError("Host name unknown. "
                         "Please fix this depending on where you store "
                         "your data .... [NOT OK]")

    args = {"baseurl": baseurl,
            "test_portion": 0.2,  # percentage of samples to take from test.
            # The left over if for train; and it will
            # be divided into actual train, and validation sets.
            # vl % of train set will be used for validation,
            # while the leftover.
            "folding": {"vl": 0.2},
            # 100-vl)/100% will be used for actual training.
            "dataset": "afad-full",
            "fold_folder": "folds/afad-full",
            "img_extension": "jpg",
            # how many times to perform the k-fold to create train, valid,
            # and test sets. The test set changes from a split to another.
            "nbr_splits": 100
            }
    split_afad(Dict2Obj(args))
    # stats_afad(Dict2Obj(args))


def do_afad_lite():
    """
    AFAD-Lite.
    :return:
    """
    # ===============
    # Reproducibility
    # ===============

    # ===========================

    reproducibility.init_seed()

    # ===========================

    # TODO: anonymize.
    if "HOST_XXX" in os.environ.keys():
        if os.environ['HOST_XXX'] == 'laptop':
            baseurl = "{}/datasets/tarball-lite".format(os.environ["EXDRIVE"])
        elif os.environ['HOST_XXX'] == 'lab':
            baseurl = "{}/datasets/tarball-lite".format(os.environ["NEWHOME"])
    elif "CC_CLUSTER" in os.environ.keys():
        baseurl = expanduser("~/workspace-sc/datasets/tarball-lite")
    else:
        raise ValueError("Host name unknown. "
                         "Please fix this depending on where you store "
                         "your data .... [NOT OK]")

    args = {"baseurl": baseurl,
            "test_portion": 0.2,  # percentage of samples to take from test.
            # The left over if for train; and it will
            # be divided into actual train, and validation sets.
            # vl % of train set will be used for validation,
            # while the leftover.
            "folding": {"vl": 0.2},
            # 100-vl)/100% will be used for actual training.
            "dataset": "afad-lite",
            "fold_folder": "folds/afad-lite",
            "img_extension": "jpg",
            # how many times to perform the k-fold to create train, valid,
            # and test sets. The test set changes from a split to another.
            "nbr_splits": 100
            }
    split_afad(Dict2Obj(args))
    # stats_afad(Dict2Obj(args))


def do_fgnet():
    """
    FGNET.
    :return:
    """
    # ===============
    # Reproducibility
    # ===============

    # ===========================

    reproducibility.init_seed()

    # ===========================
    # TODO: anonymize.
    if "HOST_XXX" in os.environ.keys():
        if os.environ['HOST_XXX'] == 'laptop':
            baseurl = "{}/datasets/FGNET".format(os.environ["EXDRIVE"])
        elif os.environ['HOST_XXX'] == 'lab':
            baseurl = "{}/datasets/FGNET".format(os.environ["NEWHOME"])
    elif "CC_CLUSTER" in os.environ.keys():
        baseurl = expanduser("~/workspace-sc/datasets/FGNET")
    else:
        raise ValueError("Host name unknown. "
                         "Please fix this depending on where you store "
                         "your data .... [NOT OK]")

    args = {"baseurl": baseurl,
            "test_portion": 0.2,  # percentage of samples to take from test.
            # The left over if for train; and it will
            # be divided into actual train, and validation sets.
            # vl % of train set will be used for validation,
            # while the leftover.
            "folding": {"vl": 0.2},
            # 100-vl)/100% will be used for actual training.
            "dataset": "fgnet",
            "fold_folder": "folds/fgnet",
            "img_extension": "JPG",
            # how many times to perform the k-fold to create train, valid,
            # and test sets. The test set changes from a split to another.
            "nbr_splits": 100
            }
    split_fgnet(Dict2Obj(args))
    stats_fgnet(Dict2Obj(args))


def do_bach_parta_2018():
    """
    BACH (PART A) 2018.
    :return:
    """
    # ===============
    # Reproducibility
    # ===============

    # ===========================

    reproducibility.set_seed()

    # ===========================

    # TODO: anonymize.
    if "HOST_XXX" in os.environ.keys():
        if os.environ['HOST_XXX'] == 'laptop':
            baseurl = "{}/datasets/ICIAR-2018-BACH-Challenge".format(
                os.environ["EXDRIVE"]
            )
        elif os.environ['HOST_XXX'] == 'lab':
            baseurl = "{}/datasets/ICIAR-2018-BACH-Challenge".format(
                os.environ["NEWHOME"]
            )
    elif "CC_CLUSTER" in os.environ.keys():
        baseurl = "{}/datasets/ICIAR-2018-BACH-Challenge".format(
            os.environ["SCRATCH"])
    else:
        raise ValueError("Host name unknown. "
                         "Please fix this depending on where you store "
                         "your data .... [NOT OK]")

    args = {"baseurl": baseurl,
            "test_portion": 0.5,  # percentage of samples to take from test.
            # The left over if for train; and it will
            # be divided into actual train, and validation sets.
            # vl/100 % of train set will be used for validation,
            # while the leftover.
            "folding": {"vl": 20},
            # 100-vl)/100% will be used for actual training.
            "name_classes": {'Normal': 0,
                             'Benign': 1,
                             'InSitu': 2,
                             'Invasive': 3},
            "dataset": "bach-part-a-2018",
            "fold_folder": "folds/bach-part-a-2018",
            "img_extension": "tif",
            "nbr_folds": 5,
            # how many times to perform the k-folds over the available train
            # samples.
            "nbr_splits": 2
            }
    create_k_folds_csv_bach_part_a(Dict2Obj(args))


def do_glas():
    """
    GlaS.

    :return:
    """
    # ===============
    # Reproducibility
    # ===============

    # ===========================

    reproducibility.set_seed()

    # ===========================

    username = getpass.getuser()
    if username == "brian":
        baseurl = "{}/datasets/GlaS-2015/Warwick QU Dataset (Released " \
                  "2016_07_08)".format(os.environ["EXDRIVE"])
    elif username == "sbelharb":
        baseurl = "{}/datasets/GlaS-2015/Warwick QU Dataset (Released " \
                  "2016_07_08)".format(os.environ["SCRATCH"])
    else:
        raise ValueError("username `{}` unknown .... [NOT OK]".format(username))

    args = {"baseurl": baseurl,
            "folding": {"vl": 20},  # 80 % for train, 20% for validation.
            "dataset": "glas",
            "fold_folder": "folds/glas-test",
            "img_extension": "bmp",
            "nbr_splits": 2  # how many times to perform the k-folds over the available train samples.
            }
    args["nbr_folds"] = math.ceil(100. / args["folding"]["vl"])
    split_valid_glas(Dict2Obj(args))


def do_Caltech_UCSD_Birds_200_2011():
    """
    Caltech-UCSD-Birds-200-2011.

    :return:
    """
    # ===============
    # Reproducibility
    # ===============

    # ===========================

    reproducibility.set_seed()

    # ===========================

    username = getpass.getuser()
    if username == "brian":
        baseurl = "{}/datasets/Caltech-UCSD-Birds-200-2011".format(
            os.environ["EXDRIVE"]
        )
    elif username == "sbelharb":
        baseurl = "{}/datasets/Caltech-UCSD-Birds-200-2011".format(
            os.environ["SCRATCH"])
    else:
        raise ValueError("username `{}` unknown .... [NOT OK]".format(username))

    args = {"baseurl": baseurl,
            "folding": {"vl": 20},  # 80 % for train, 20% for validation.
            "dataset": "Caltech-UCSD-Birds-200-2011",
            "fold_folder": "folds/Caltech-UCSD-Birds-200-2011",
            "img_extension": "bmp",
            "nbr_splits": 2,  # how many times to perform the k-folds over
            # the available train samples.
            "path_encoding": "folds/Caltech-UCSD-Birds-200-2011/encoding-origine.yaml",
            "nbr_classes": None  # Keep only 5 random classes. If you want
            # to use the entire dataset, set this to None.
            }
    args["nbr_folds"] = math.ceil(100. / args["folding"]["vl"])
    split_valid_Caltech_UCSD_Birds_200_2011(Dict2Obj(args))


def do_Oxford_flowers_102():
    """
    Oxford-flowers-102.
    The train/valid/test sets are already provided.

    :return:
    """
    # ===============
    # Reproducibility
    # ===============

    # ===========================

    reproducibility.set_seed()

    # ===========================

    username = getpass.getuser()
    if username == "brian":
        baseurl = "{}/datasets/Oxford-flowers-102".format(os.environ["EXDRIVE"])
    elif username == "sbelharb":
        baseurl = "{}/datasets/Oxford-flowers-102".format(os.environ["SCRATCH"])
    else:
        raise ValueError("username `{}` unknown .... [NOT OK]".format(username))

    args = {"baseurl": baseurl,
            "dataset": "Oxford-flowers-102",
            "fold_folder": "folds/Oxford-flowers-102",
            "img_extension": "jpg",
            "path_encoding": "folds/Oxford-flowers-102/encoding-origine.yaml"
            }
    # Convert masks into binary masks.
    create_bin_mask_Oxford_flowers_102(Dict2Obj(args))
    reproducibility.set_seed()
    split_Oxford_flowers_102(Dict2Obj(args))

    # Find min max size.
    def find_stats(argsx):
        """

        :param argsx:
        :return:
        """
        minh, maxh, minw, maxw = None, None, None, None
        baseurl = argsx.baseurl
        fin = find_files_pattern(join(baseurl, 'jpg'), '*.jpg')
        print("Computing stats from {} dataset ...".format(argsx.dataset))
        for f in tqdm.tqdm(fin, ncols=80, total=len(fin)):
            w, h = Image.open(f, 'r').convert('RGB').size
            if minh is None:
                minh = h
                maxh = h
                minw = w
                maxw = w
            else:
                minh = min(minh, h)
                maxh = max(maxh, h)
                minw = min(minw, w)
                maxw = max(maxw, w)

        print('Stats {}:\n'
              'min h: {} \n'
              'max h: {} \n'
              'min w: {} \n'
              'max w: {} \n'.format(argsx.dataset, minh, maxh, minw, maxw))

    find_stats(Dict2Obj(args))


if __name__ == "__main__":
    check_if_allow_multgpu_mode()

    # ============== CREATE FOLDS OF HISTORICAL COLOR IMAGE DATASET
    # do_historical_color_image_decade()

    # ============== CREATE FOLDS OF AFAD-Full DATASET
    do_afad_full()

    # ============== CREATE FOLDS OF AFAD-Lite DATASET
    do_afad_lite()
    # ============== CREATE FOLDS OF FGNET DATASET
    # do_fgnet()
    # ============== CREATE FOLDS OF BACH (PART A) 2018 DATASET
    # do_bach_parta_2018()

    # ============== CREATE FOLDS OF GlaS DATASET
    # do_glas()

    # ============== CREATE FOLDS OF Caltech-UCSD-Birds-200-2011 DATASET
    # do_Caltech_UCSD_Birds_200_2011()

    # ============== CREATE FOLDS OF Oxford-flowers-102 DATASET
    # do_Oxford_flowers_102()
