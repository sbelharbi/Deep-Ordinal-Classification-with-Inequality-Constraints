import csv
from os.path import join
import collections
import copy
import warnings
import datetime as dt

import PIL
from PIL import Image
import numpy as np
import tqdm

import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F


import reproducibility


__all__ = ["PhotoDataset"]


def default_collate(batch):
    """
    Override
    https://pytorch.org/docs/stable/_modules/torch/utils/data/
    dataloader.html#DataLoader

    Reference:
    def default_collate(batch) at
    https://pytorch.org/docs/stable/_modules/torch/utils/data/
    dataloader.html#DataLoader
    https://discuss.pytorch.org/t/
    how-to-create-a-dataloader-with-variable-size-input/8278/3
    https://github.com/pytorch/pytorch/issues/1512

    We need our own collate function that wraps things up
    (imge, mask, label, size).

    In this setup,  batch is a list of tuples (the result of calling:
    img, mask, label = PhotoDataset[i].
    The output of this function is four elements:
        * data: a pytorch tensor of size (batch_size, c, h, w) of float32 .
        Each sample is a tensor of shape (c, h_, w_) that represents a cropped
        patch from an image (or the entire image) where: c is the depth of the
        patches (since they are RGB, so c=3),  h is the height of the patch,
        and w_ is the its width.
        * mask[optional=None]: a list of pytorch tensors of size (batch_size,
        1, h, w) full of 1 and 0. The mask of the ENTIRE image (no cropping is
        performed). Images does not have the same size, and the same thing
        goes for the masks. Therefore, we can't put the masks in one tensor.
        If there is no mask in the ground truth, return None.
        * target: a vector (pytorch tensor) of length batch_size of type
        torch.LongTensor containing the image-level labels.
    :param batch: list of tuples (img, mask [optional=None], label)
    :return: 3 elements: tensor data, list of tensors of masks, tensor of labels.
    """
    data = torch.stack([item[0] for item in batch])
    if batch[0][0] is not None:
        mask = [item[1] for item in batch]  # each element is of size (1, h, w).
    else:
        mask = None
    target = torch.LongTensor([item[2] for item in batch])

    return data, mask, target


def csv_loader(fname, rootpath):
    """
    Read a *.csv file. Each line contains:
     * relative path to an image.
     * [optional] relative path to the mask of the object of interest.
     * image class (str).

    :param fname: Path to the *.csv file.
    :param rootpath: The root path to the folders of the images.
    :return: List of elements.
    Each element is the path to an image: image path, mask path [optional],
    class name.
    """
    with open(fname, 'r') as f:
        nbr_fields = 2
        for rowx in csv.reader(f):
            nbr_fields = len(rowx)
            break
        if nbr_fields == 2:
            out = [
                [join(rootpath, row[0]), row[1]]
                for row in csv.reader(f)
            ]
        elif nbr_fields == 3:
            if nbr_fields == 2:
                out = [
                    [join(rootpath, row[0]), join(rootpath, row[1]), row[2]]
                    for row in csv.reader(f)
                ]

    return out


class MyDataParallel(torch.nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def forward(self, *inputs, **kwargs):
        """
        The exact same as in Pytorch.
        We use it for debugging.
        :param inputs:
        :param kwargs:
        :return:
        """
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        tx = dt.datetime.now()
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        # print("Scattering took {}".format(dt.datetime.now() - tx))
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        tx = dt.datetime.now()
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        # print("Replicating took {}".format(dt.datetime.now() - tx))
        tx = dt.datetime.now()
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        # print("Gathering took {}".format(dt.datetime.now() - tx))
        return self.gather(outputs, self.output_device)


class MyRandomCropper(transforms.RandomCrop):
    """
    Crop the given PIL Image at a random location.

    Class inherits from transforms.RandomCrop().
    It does exactly the same thing, except, it returns the coordinates of
    along with the crop.
    """
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
            Coordinates of the crop: tuple (i, j, h, w).
        """
        if self.padding > 0:
            img = F.pad(img, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

        i, j, h, w = self.get_params(img, self.size)

        return TF.crop(img, i, j, h, w), (i, j, h, w)


class PhotoDataset(Dataset):
    """
    Class that overrides torch.utils.data.Dataset.
    """
    def __init__(self, data, dataset_name, name_classes, transform_tensor,
                 set_for_eval, transform_img=None, resize=None,
                 crop_size=None, padding_size=None, padding_mode="reflect",
                 up_scale_small_dim_to=None,
                 do_not_save_samples=False, ratio_scale_patch=1.,
                 for_eval_flag=False):
        """
        :param data: A list of str absolute paths of the images of dataset.
               In this case, no preprocessing will be used (such brightness
               standardization, ...). Raw data will be used
               directly.
        :param dataset_name: str, name of the dataset.
        :param name_classes: dict, {"classe_name": int}.
        :param transform_tensor: a composition of transforms that performs over
               torch.tensor: torchvision.transforms.Compose(). or None.
        :param set_for_eval: True/False. Used to entirely prepare the data for
               evaluation by performing all the necessary steps to get the
               data ready for input to the model. Useful for the evaluation
               datasets such as the validation set or the test test. If True we
               do all that, else the preparation of the data is done when
               needed. If  dataset is LARGE AND you inscrease the size of the
               samples through a processing step (upscaling for instance), we
               recommend to set this to False since you will need a large
               memory. In this case, the validation will be slow (you can
               increase the number of workers if you use a batch size of 1).
        :param transform_img: a composition of transforms that performs over
               images: torchvision.transforms.Compose(). or None.
        :param resize: int, or sequence of two int (w, h), or None. The size
               to which the original image is resized. If None, the original
               image is used. (needed only when data is a list).
        :param crop_size: tuple of int (h, w). Size of the cropped patches.
               If None, no cropping is done, and the entire image is used
               (such the case in validation).
        :param padding_size: (h%, w%), how much to pad (top/bottom) and
               (left/right) of the ORIGINAL IMAGE. h, w are percentages.
        :param padding_mode: str, accepted padding mode (
               https://pytorch.org/docs/stable/torchvision/transforms.html
               #torchvision.transforms.functional.pad)
        :param up_scale_small_dim_to: int or None. If not None, we upscale the
               small dimension (height or width) to this value (then compute
               the upscale ration r). Then, upscale the other dimension to a
               proportional value (using the ratio r). This is helpful when the
               images have small size such as in the dataset
               Caltech-UCSD-Birds-200-2011. Due to the depth of the model,
               small images may 'disappear' or provide a very small
               attention map.
        :param do_not_save_samples: Bool. If True, we do not save samples in
               memory. The default behavior of the code is to preload the
               samples, and save them in memory to speedup access and to avoid
               disc access. However, this may be impractical when dealing with
               large dataset during the final processing (evaluation). It is
               not necessary to keep the samples of the dataset in the memory
               once they are processed. Consequences to this boolean flag: If
               it is True, we do not preload sample (read from disc), AND once a
               sample is loaded, it is not stored. There few things that we
               save: 1. The size of the sample (h, w). in
               self.original_images_size. We remind that this flag is useful
               only at the end of the training when you want to evaluate on a
               set (train, valid, test). In this case, there is no need to store
               anything. If the dataset is large, this will cause memory
               overflow (in case you run your code on a server with reserved
               amount of memory). If you set this flag to True, use 0 workers
               for the dataloader, since we will be processing the samples
               sequentially, and we want to avoid to load a sample ahead
               (no point of doing that).
               This has another usefullness when dealing with large train
               dataset and using multiprocessing. You will need a huge memory to
               work. To keep using myltiprocessing we recommend setting this
               to True so you do not save the image in memory (
               multiprocessing duplicates the main process memory), but you
               read from disc all the time, but you need extremely fast disc
               access. It a compromise either you have an insane size of RAM
               per job or you have a fast disc access.
        :param ratio_scale_patch: the ratio to which the cropped patch is
               scaled. during evaluation, original images are also rescaled
               using this ratio. if you want to keep the cropped patch as it
               is, set this variable to 1.
        :param for_eval_flag: bool. Set this to true of the dataset is
               intended for evaluation. This is different from
               `set_for_eval`. The latter affects how we process the images,
               while this variable just tells us if this dataset will be used
               evaluation or not.
        """
        assert isinstance(
            data, list), "`data` is supposed to be of type: list. " \
                         "Found {}".format(type(data))

        msg = "must 0 < `ration_scale_patch` <=1. found {} ...[NOT " \
              "OK]".format(ratio_scale_patch)
        assert 0. < ratio_scale_patch <= 1., msg

        self.ratio_scale_patch = ratio_scale_patch
        # convert mask to tensor.
        self.to_tensor = transforms.Compose([transforms.ToTensor()])

        self.set_for_eval = set_for_eval
        self.set_for_eval_backup = set_for_eval
        self.for_eval_flag = for_eval_flag
        self.name_classes = name_classes
        self.up_scale_small_dim_to = up_scale_small_dim_to
        self.do_not_save_samples = do_not_save_samples

        allowed_datasets = ["bach-part-a-2018",
                            "fgnet",
                            "afad-lite",
                            "afad-full",
                            "historical-color-image-decade",
                            "Caltech-UCSD-Birds-200-2011",
                            "Oxford-flowers-102"
                            ]
        msg = "dataset_name = {} unsupported. Please double check. We do " \
              "some operations that are dataset dependent. So, you may " \
              "need to do these operations on your own (mask binarization, " \
              "...). Exiting .... [NOT OK]".format(dataset_name)
        assert dataset_name in allowed_datasets, msg

        # only medical datasets are allowed to be padded since it makes sens.
        if dataset_name != "bach-part-a-2018":
            assert padding_size is None, "We do not support padding " \
                                         "train/test for data other than " \
                                         "`bach-part-a-2018` set."

        self.dataset_name = dataset_name
        self.samples = data

        self.seeds = None
        self.set_up_new_seeds()  # set up seeds for the initialization.

        self.transform_img = transform_img
        self.transform_tensor = transform_tensor
        self.resize = None
        if resize:
            if isinstance(resize, int):
                self.resize = (resize, resize)
            elif isinstance(self.resize, collections.Sequence):
                self.resize = resize

        if crop_size:
            self.randomCropper = MyRandomCropper(size=crop_size, padding=0)
        else:
            self.randomCropper = None

        self.padding_size = padding_size
        if padding_size:
            msg = "You asked for padding, but you didn't specify the " \
                  "padding mode. Accepted modes can be found in " \
                  "https://pytorch.org/docs/stable/torchvision/transforms." \
                  "html#torchvision.transforms.functional.pad"
            assert padding_mode is not None, msg
        self.padding_mode = padding_mode
        self.n = len(self.samples)
        self.images = []
        self.original_images_size = [None for _ in range(len(self))]
        self.absolute_paths_imgs = []
        self.absolute_paths_masks = []

        self.with_segm = False  # no masks are provided with g.turth.
        assert len(self.samples[0]) in [2, 3], "Your data is weird. Each row" \
                                               "either has: img, cl or img," \
                                               "msk, cl. We found {} ...." \
                                               "[NOT OK]".format(
            len(self.samples[0]))

        if len(self.samples[0]) == 3:
            self.with_segm = True

        if self.with_segm:
            for path_img, path_mask, _ in self.samples:
                self.absolute_paths_imgs.append(path_img)
                self.absolute_paths_masks.append(path_mask)
        else:
            for path_img, _ in self.samples:
                self.absolute_paths_imgs.append(path_img)

        self.labels = []
        self.masks = []
        self.preloaded = False

        if not do_not_save_samples:
            self.preload_images()

        self.inputs_ready = []
        self.labels_ready = []
        self.masks_ready = []

        if self.set_for_eval:
            self.set_ready_eval()

    def set_up_new_seeds(self):
        """
        Set up new seed for each sample.
        :return:
        """
        self.seeds = self.get_new_seeds()

    def get_new_seeds(self):
        """
        Generate a seed per sample.
        :return:
        """
        return np.random.randint(0, 100000, len(self))

    def get_original_input_img(self, i):
        """
        Returns the original input image read from disc.
        :param i: index of the sample.
        :return:
        """
        # TODO: deal with gray images.
        return Image.open(self.samples[i][0], "r").convert("RGB")

    def get_path_input_img(self, i):
        """
        Return the path of the input image.
        :param i: int, index of the sample.
        :return: str, path of the input image.
        """
        return self.samples[i][0]

    def get_original_input_mask(self, i):
        """
        Returns the original input mask read from disc.
        :param i: index of the sample.
        :return:
        """
        assert self.with_segm, "The dataset {} does not have any mask" \
                               "ground truth. But, you seem to " \
                               "want to load a mask. This is weird." \
                               ".... [NOT OK]".format(self.dataset_name)

        mask = Image.open(self.samples[i][1], "r").convert("L")

        # Caltech-UCSD-Birds-200-2011: a pixel belongs to the mask if its
        # value > 255/2. (an image is annotated
        # by many workers. If more than half of the workers agreed on the pixel
        # to be a bird, we consider that pixel as a bird.

        # Oxford-flowers-102: a pixel belongs to the mask if its value > 0.
        # The mask has only {0, 255} as values. The new binary mask will
        # contain only {0, 1} values where 0 is the background and 1 is the
        # foreground.
        mask_np = np.array(mask)
        if self.dataset_name == "Caltech-UCSD-Birds-200-2011":
            mask_np = (mask_np > (255 / 2.)).astype(np.uint8)
        elif self.dataset_name == 'Oxford-flowers-102':
            mask_np = (mask_np != 0).astype(np.uint8)
        else:
            raise ValueError("Dataset name {} unsupported. Exiting .... "
                             "[NOT OK]".format(self.dataset_name))

        mask = Image.fromarray(mask_np * 255, mode="L")

        return mask

    def get_original_input_label_int(self, i):
        """
        Returns the input image-level label.
        :param i: index of the sample.
        :return:
        """
        if self.with_segm:
            label_str = self.samples[i][2]
        else:
            label_str = self.samples[i][1]
        return self.name_classes[label_str]

    def load_sample_i(self, i):
        """
        Read from disc sample number i.
        :param i: index of the sample to load.
        :return: image, mask[optional], label.
        """
        img = self.get_original_input_img(i)
        if self.with_segm:
            mask = self.get_original_input_mask(i)
        else:
            mask = None
        label = self.get_original_input_label_int(i)

        self.original_images_size[i] = img.size

        # This if is not used.
        if self.resize:
            img = img.resize(self.resize)
            if self.with_segm:
                mask = mask.resize(self.resize)

        return img, mask, label

    def preload_images(self):
        """
        Preload images/masks[optional]/labels.
        :return:
        """
        print("Preloading the images of `{}` dataset. "
              "This may take some time ... [OK]".format(self.dataset_name))
        tx = dt.datetime.now()

        for i in tqdm.tqdm(range(self.n), ncols=80, total=self.n):
            img, mask, label = self.load_sample_i(i)

            self.images.append(img)
            self.masks.append(mask)
            self.labels.append(label)

        self.preloaded = True
        print("{} has successfully loaded the images with {} samples in"
              " {} .... [OK]".format(
               self.dataset_name, self.n, dt.datetime.now() - tx))

    @staticmethod
    def get_upscaled_dims(w, h, up_scale_small_dim_to):
        """
        Compute the upscaled dimensions using the size `up_scale_small_dim_to`.

        :param w:
        :param h:
        :param up_scale_small_dim_to:
        :return: w, h: the width and the height upscale (with preservation of
         the ratio).
        """
        if up_scale_small_dim_to is None:
            return w, h

        s = up_scale_small_dim_to
        if h < s:
            if h < w:  # find the maximum ratio to scale.
                r = (s / h)
            else:
                r = (s / w)
        elif w < s:  # find the maximum ratio to scale.
            if w < h:
                r = (s / w)
            else:
                r = (s / h)
        else:
            r = 1  # no upscaling since both dims are higher or equal to
            # the min (s).
        h_, w_ = int(h * r), int(w * r)

        return w_, h_

    def set_ready_eval(self):
        """
        Prepare the data for evaluation [Called ONLY ONCE].

        This function is useful when this class is instantiated over an
        evaluation set with no randomness, such as the valid set or the test
        set.

        The idea is to prepare the data by performing all the necessary steps
        until we arrive to the final form of the input of the model.

        This will avoid doing all the steps every time self.__getitem__() is
        called.

        :return:
        """
        assert self.set_for_eval, "Something wrong. You didn't ask to set " \
                                  "the data ready for evaluation, " \
                                  "but here we are .... [NOT OK]"
        assert self.images is not None, "self.images is not ready yet. " \
                                        "Re-check .... [NOT OK]"
        assert self.masks is not None, "self.masks is not ready yet. " \
                                       "Re-check ... [NOT OK]"
        assert self.labels is not None, "self.labels is not ready yet. " \
                                        "Re-check ... [NOT OK]"

        print("Setting `{}` this dataset for evaluation. "
              "This may take some time ... [OK]".format(self.dataset_name))
        tx = dt.datetime.now()

        # Turn off momentarily self.set_for_eval.
        self.set_for_eval = False

        for i in tqdm.tqdm(range(len(self.images)), ncols=80, total=self.n):
            sample, mask, target = self.__getitem__(i)
            self.inputs_ready.append(sample)
            self.masks_ready.append(mask)
            self.labels_ready.append(target)

        # Turn self.set_for_eval back on.
        self.set_for_eval = True
        # Now that we preloaded everything, we need to remove self.images,
        # self.masks, to preserve space!!!
        # We keep self.labels. We need it!!! and it does not take much space!
        del self.images
        del self.masks
        del self.labels

        print("`{}` dataset has been set ready for evaluation with"
              " `{}` samples ready to go in {} .... [OK]".format(
                self.dataset_name, self.n, dt.datetime.now() - tx))

    @staticmethod
    def get_padding(s, c):
        """
        Find out how much padding in both sides (left/right) or (top/bottom)
        is required.
        :param s: height or width of the image.
        :param c: constant such as after padding we will have: s % c = 0.
        :return: pad1, pad2. Padding in both sides.
        """
        assert isinstance(s, int) and isinstance(c, int), "s, and c must " \
                                                          "be integers ...." \
                                                          " [NOT OK]"

        if s % c == 0:
            return 0, 0
        leftover = c - s % c
        if leftover % 2 == 0:
            return int(leftover / 2), int(leftover / 2)
        else:
            return int(leftover / 2), leftover - int(leftover / 2)

    def __getitem__(self, index):
        """
        Return one sample and its label and extra information that we need
        later.

        :param index: int, the index of the sample within the whole dataset.
        :return: sample: pytorch.tensor of size (1, C, H, W) and datatype
                 torch.FloatTensor. Where C is the number of
                 color channels (=3), and H is the height of the patch, and W
                 is its width.
                 mask: PIL.Image.Image, the mask of the regions of interest.
                 or None if there is no ground truth mask.
                 label: int, the label of the sample.
        """
        # Force seeding: a workaround to deal with reproducibility when suing
        # different number of workers if want to
        # preserve the reproducibility. Each sample has its won seed.
        reproducibility.reset_seed(self.seeds[index])

        if self.set_for_eval:
            error_msg = "Something wrong. You didn't ask to set the data " \
                        "ready for evaluation, but here we are " \
                        ".... [NOT OK]"
            assert self.inputs_ready is not None and \
                self.labels_ready is not None, error_msg
            img = self.inputs_ready[index]
            mask = self.masks_ready[index]
            target = self.labels_ready[index]

            return img, mask, target

        if self.do_not_save_samples:
            img, mask, target = self.load_sample_i(index)
        else:
            assert self.preloaded, "Sorry, you need to preload the data " \
                                   "first .... [NOT OK]"
            img, mask, target = self.images[index], self.masks[index], \
                self.labels[index]

        # Upscale on the fly. Sorry, this may add an extra time, but, we do
        # not want to save in memory upscaled images!!!! it takes a lot of
        # space, especially for large datasets. So, compromise? upscale only
        # when necessary.
        # check if we need to upscale the image.
        # Useful for Caltech-UCSD-Birds-200-2011 for instance.
        if self.up_scale_small_dim_to is not None:
            w, h = img.size
            w_up, h_up = self.get_upscaled_dims(
                w, h, self.up_scale_small_dim_to)
            # make a resized copy.
            img = img.resize((w_up, h_up), resample=PIL.Image.BILINEAR)

        # Upscale the image: only for Caltech-UCSD-Birds-200-2011 and similar
        # datasets.

        # crop a patch (training only). Do not crop for evaluation.
        if self.randomCropper:
            msg = "Something's wrong. This is expected to be False." \
                  "We do not crop for evaluation."
            assert not self.for_eval_flag, msg
            # Padding.
            if self.padding_size:
                w, h = img.size
                ph, pw = self.padding_size
                padding = (int(pw * w), int(ph * h))
                img = TF.pad(img, padding=padding,
                             padding_mode=self.padding_mode)
                # just for tracking.
                if mask is not None:
                    mask = TF.pad(mask, padding=padding,
                                  padding_mode=self.padding_mode)

            img, (i, j, h, w) = self.randomCropper(img)
            # print("Dadaloader Index {} i  {}  j {} seed {}".format(
            # index, i, j, self.seeds[index]))
            # crop the mask just for tracking. Not used for actual training.
            if mask is not None:
                mask = TF.crop(mask, i, j, h, w)

            if self.ratio_scale_patch < 1.:
                img = img.resize((int(img.size[0] * self.ratio_scale_patch),
                                  int(img.size[1] * self.ratio_scale_patch)))

        # rescale the image with the same ration that we use to rescale the
        # cropped patches.
        if self.for_eval_flag and (self.ratio_scale_patch < 1.):
            img = img.resize((int(img.size[0] * self.ratio_scale_patch),
                              int(img.size[1] * self.ratio_scale_patch)))

        # just for training: do not transform the mask (since it is not used).
        if self.transform_img:
            img = self.transform_img(img)

        # just for training: do not transform the mask (since it is not used).
        if self.transform_tensor:
            img = self.transform_tensor(img)

        # Prepare the mask to be used on GPU to compute Dice index.
        if mask is not None:
            mask = np.array(mask, dtype=np.float32) / 255.  # full of 0 and 1.
            # mak the mask with shape (h, w, 1):
            mask = self.to_tensor(np.expand_dims(mask, axis=-1))

        return img, mask, target

    def __len__(self):
        return len(self.samples)