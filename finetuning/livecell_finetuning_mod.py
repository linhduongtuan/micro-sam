import argparse
import os

import torch
import torch_em
from torch_em.data.datasets import get_livecell_loader

#import micro_sam.training as sam_training
#from micro_sam.util import export_custom_sam_model


import os
import random
import time
from typing import Optional

import numpy as np
import torch
import torch_em
from torch_em.trainer.logger_base import TorchEmLogger
from torchvision.utils import make_grid

#from ..prompt_generators import IterativePromptGenerator, PromptGeneratorBase
from typing import Any, Dict, List, Optional, Union

import torch
from segment_anything.modeling import Sam
from torch import nn
from torch.nn import functional as F

import os
from typing import List, Optional, Union

import numpy as np

#from ..prompt_generators import PointAndBoxPromptGenerator
#from ..util import (
#    _get_device,
#    get_centers_and_bounding_boxes,
#    get_sam_model,
#    segmentation_to_one_hot,
#)
#from .trainable_sam import TrainableSAM
from typing import List, Optional, Tuple

import numpy as np
import torch
from kornia import morphology


class PromptGeneratorBase:
    """PromptGeneratorBase is an interface to implement specific prompt generators.
    """
    def __call__(
            self,
            segmentation: torch.Tensor,
            prediction: Optional[torch.Tensor] = None,
            bbox_coordinates: Optional[List[tuple]] = None,
            center_coordinates: Optional[List[np.ndarray]] = None
    ) -> Tuple[
        Optional[torch.Tensor],  # the point coordinates
        Optional[torch.Tensor],  # the point labels
        Optional[torch.Tensor],  # the bounding boxes
        Optional[torch.Tensor],  # the mask prompts
    ]:
        """Return the point prompts given segmentation masks and optional other inputs.

        Args:
            segmentation: The object masks derived from instance segmentation groundtruth.
                Expects a float tensor of shape NUM_OBJECTS x 1 x H x W.
                The first axis corresponds to the binary object masks.
            prediction: The predicted object masks corresponding to the segmentation.
                Expects the same shape as the segmentation
            bbox_coordinates: Precomputed bounding boxes for the segmentation.
                Expects a list of length NUM_OBJECTS.
            center_coordinates: Precomputed center coordinates for the segmentation.
                Expects a list of length NUM_OBJECTS.

        Returns:
            The point prompt coordinates. Int tensor of shape NUM_OBJECTS x NUM_POINTS x 2.
                The point coordinates are retuned in XY axis order. This means they are reversed compared
                to the standard YX axis order used by numpy.
            The point prompt labels. Int tensor of shape NUM_OBJECTS x NUM_POINTS.
            The box prompts. Int tensor of shape NUM_OBJECTS x 4.
                The box coordinates are retunred as MIN_X, MIN_Y, MAX_X, MAX_Y.
            The mask prompts. Float tensor of shape NUM_OBJECTS x 1 x H' x W'.
                With H' = W'= 256.
        """
        raise NotImplementedError("PromptGeneratorBase is just a class template. \
                                  Use a child class that implements the specific generator instead")


class PointAndBoxPromptGenerator(PromptGeneratorBase):
    """Generate point and/or box prompts from an instance segmentation.

    You can use this class to derive prompts from an instance segmentation, either for
    evaluation purposes or for training Segment Anything on custom data.
    In order to use this generator you need to precompute the bounding boxes and center
    coordiantes of the instance segmentation, using e.g. `util.get_centers_and_bounding_boxes`.

    Here's an example for how to use this class:
    ```python
    # Initialize generator for 1 positive and 4 negative point prompts.
    prompt_generator = PointAndBoxPromptGenerator(1, 4, dilation_strength=8)

    # Precompute the bounding boxes for the given segmentation
    bounding_boxes, _ = util.get_centers_and_bounding_boxes(segmentation)

    # generate point prompts for the objects with ids 1, 2 and 3
    seg_ids = (1, 2, 3)
    object_mask = np.stack([segmentation == seg_id for seg_id in seg_ids])[:, None]
    this_bounding_boxes = [bounding_boxes[seg_id] for seg_id in seg_ids]
    point_coords, point_labels, _, _ = prompt_generator(object_mask, this_bounding_boxes)
    ```

    Args:
        n_positive_points: The number of positive point prompts to generate per mask.
        n_negative_points: The number of negative point prompts to generate per mask.
        dilation_strength: The factor by which the mask is dilated before generating prompts.
        get_point_prompts: Whether to generate point prompts.
        get_box_prompts: Whether to generate box prompts.
    """
    def __init__(
        self,
        n_positive_points: int,
        n_negative_points: int,
        dilation_strength: int,
        get_point_prompts: bool = True,
        get_box_prompts: bool = False
    ) -> None:
        self.n_positive_points = n_positive_points
        self.n_negative_points = n_negative_points
        self.dilation_strength = dilation_strength
        self.get_box_prompts = get_box_prompts
        self.get_point_prompts = get_point_prompts

        if self.get_point_prompts is False and self.get_box_prompts is False:
            raise ValueError("You need to request box prompts, point prompts or both.")

    def _sample_positive_points(self, object_mask, center_coordinates, coord_list, label_list):
        if center_coordinates is not None:
            # getting the center coordinate as the first positive point (OPTIONAL)
            coord_list.append(tuple(map(int, center_coordinates)))  # to get int coords instead of float

            # getting the additional positive points by randomly sampling points
            # from this mask except the center coordinate
            n_positive_remaining = self.n_positive_points - 1

        else:
            # need to sample "self.n_positive_points" number of points
            n_positive_remaining = self.n_positive_points

        if n_positive_remaining > 0:
            object_coordinates = torch.where(object_mask)
            n_coordinates = len(object_coordinates[0])

            # randomly sampling n_positive_remaining_points from these coordinates
            indices = np.random.choice(
                n_coordinates, size=n_positive_remaining,
                # Allow replacing if we can't sample enough coordinates otherwise
                replace=True if n_positive_remaining > n_coordinates else False,
            )
            coord_list.extend([
                [object_coordinates[0][idx], object_coordinates[1][idx]] for idx in indices
            ])

        label_list.extend([1] * self.n_positive_points)
        assert len(coord_list) == len(label_list) == self.n_positive_points
        return coord_list, label_list

    def _sample_negative_points(self, object_mask, bbox_coordinates, coord_list, label_list):
        if self.n_negative_points == 0:
            return coord_list, label_list

        # getting the negative points
        # for this we do the opposite and we set the mask to the bounding box - the object mask
        # we need to dilate the object mask before doing this: we use kornia.morphology.dilation for this
        dilated_object = object_mask[None, None]
        for _ in range(self.dilation_strength):
            dilated_object = morphology.dilation(dilated_object, torch.ones(3, 3), engine="convolution")
        dilated_object = dilated_object.squeeze()

        background_mask = torch.zeros(object_mask.shape, device=object_mask.device)
        _ds = self.dilation_strength
        background_mask[max(bbox_coordinates[0] - _ds, 0): min(bbox_coordinates[2] + _ds, object_mask.shape[-2]),
                        max(bbox_coordinates[1] - _ds, 0): min(bbox_coordinates[3] + _ds, object_mask.shape[-1])] = 1
        background_mask = torch.abs(background_mask - dilated_object)

        # the valid background coordinates
        background_coordinates = torch.where(background_mask)
        n_coordinates = len(background_coordinates[0])

        # randomly sample the negative points from these coordinates
        indices = np.random.choice(
            n_coordinates, replace=False,
            size=min(self.n_negative_points, n_coordinates)  # handles the cases with insufficient bg pixels
        )
        coord_list.extend([
            [background_coordinates[0][idx], background_coordinates[1][idx]] for idx in indices
        ])
        label_list.extend([0] * len(indices))

        return coord_list, label_list

    def _ensure_num_points(self, object_mask, coord_list, label_list):
        num_points = self.n_positive_points + self.n_negative_points

        # fill up to the necessary number of points if we did not sample enough of them
        if len(coord_list) != num_points:
            # to stay consistent, we add random points in the background of an object
            # if there's no neg region around the object - usually happens with small rois
            needed_points = num_points - len(coord_list)
            more_neg_points = torch.where(object_mask == 0)
            indices = np.random.choice(len(more_neg_points[0]), size=needed_points, replace=False)

            coord_list.extend([
                (more_neg_points[0][idx], more_neg_points[1][idx]) for idx in indices
            ])
            label_list.extend([0] * needed_points)

        assert len(coord_list) == len(label_list) == num_points
        return coord_list, label_list

    # Can we batch this properly?
    def _sample_points(self, segmentation, bbox_coordinates, center_coordinates):
        all_coords, all_labels = [], []

        center_coordinates = [None] * len(segmentation) if center_coordinates is None else center_coordinates
        for object_mask, bbox_coords, center_coords in zip(segmentation, bbox_coordinates, center_coordinates):
            coord_list, label_list = [], []
            coord_list, label_list = self._sample_positive_points(
                object_mask[0], center_coords, coord_list, label_list
            )
            coord_list, label_list = self._sample_negative_points(
                object_mask[0], bbox_coords, coord_list, label_list
            )
            coord_list, label_list = self._ensure_num_points(object_mask[0], coord_list, label_list)

            all_coords.append(coord_list)
            all_labels.append(label_list)

        return all_coords, all_labels

    # TODO make compatible with exact same input shape
    def __call__(
        self,
        segmentation: torch.Tensor,
        bbox_coordinates: List[Tuple],
        center_coordinates: Optional[List[np.ndarray]] = None,
        **kwargs,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        None
    ]:
        """Generate the prompts for one object in the segmentation.

        Args:
            segmentation: Instance segmentation masks .
            bbox_coordinates: The precomputed bounding boxes of particular object in the segmentation.
            center_coordinates: The precomputed center coordinates of particular object in the segmentation.
                If passed, these coordinates will be used as the first positive point prompt.
                If not passed a random point from within the object mask will be used.

        Returns:
            Coordinates of point prompts. Returns None, if get_point_prompts is false.
            Point prompt labels. Returns None, if get_point_prompts is false.
            Bounding box prompts. Returns None, if get_box_prompts is false.
        """
        if self.get_point_prompts:
            coord_list, label_list = self._sample_points(segmentation, bbox_coordinates, center_coordinates)
            # change the axis convention of the point coordinates to match the expected coordinate order of SAM
            coord_list = np.array(coord_list)[:, :, ::-1].copy()
            coord_list = torch.from_numpy(coord_list)
            label_list = torch.tensor(label_list)
        else:
            coord_list, label_list = None, None

        if self.get_box_prompts:
            # change the axis convention of the point coordinates to match the expected coordinate order of SAM
            bbox_list = np.array(bbox_coordinates)[:, [1, 0, 3, 2]]
            bbox_list = torch.from_numpy(bbox_list)
        else:
            bbox_list = None

        return coord_list, label_list, bbox_list, None


class IterativePromptGenerator(PromptGeneratorBase):
    """Generate point prompts from an instance segmentation iteratively.
    """
    def _get_positive_points(self, pos_region, overlap_region):
        positive_locations = [torch.where(pos_reg) for pos_reg in pos_region]
        # we may have objects without a positive region (= missing true foreground)
        # in this case we just sample a point where the model was already correct
        positive_locations = [
            torch.where(ovlp_reg) if len(pos_loc[0]) == 0 else pos_loc
            for pos_loc, ovlp_reg in zip(positive_locations, overlap_region)
        ]
        # we sample one location for each object in the batch
        sampled_indices = [np.random.choice(len(pos_loc[0])) for pos_loc in positive_locations]
        # get the corresponding coordinates (Note that we flip the axis order here due to the expected order of SAM)
        pos_coordinates = [
            [pos_loc[-1][idx], pos_loc[-2][idx]] for pos_loc, idx in zip(positive_locations, sampled_indices)
        ]

        # make sure that we still have the correct batch size
        assert len(pos_coordinates) == pos_region.shape[0]
        pos_labels = [1] * len(pos_coordinates)

        return pos_coordinates, pos_labels

    # TODO get rid of this looped implementation and use proper batched computation instead
    def _get_negative_points(self, negative_region_batched, true_object_batched):
        device = negative_region_batched.device

        negative_coordinates, negative_labels = [], []
        for neg_region, true_object in zip(negative_region_batched, true_object_batched):

            tmp_neg_loc = torch.where(neg_region)
            if torch.stack(tmp_neg_loc).shape[-1] == 0:
                tmp_true_loc = torch.where(true_object)
                x_coords, y_coords = tmp_true_loc[1], tmp_true_loc[2]
                bbox = torch.stack([torch.min(x_coords), torch.min(y_coords),
                                    torch.max(x_coords) + 1, torch.max(y_coords) + 1])
                bbox_mask = torch.zeros_like(true_object).squeeze(0)

                custom_df = 3  # custom dilation factor to perform dilation by expanding the pixels of bbox
                bbox_mask[max(bbox[0] - custom_df, 0): min(bbox[2] + custom_df, true_object.shape[-2]),
                          max(bbox[1] - custom_df, 0): min(bbox[3] + custom_df, true_object.shape[-1])] = 1
                bbox_mask = bbox_mask[None].to(device)

                background_mask = torch.abs(bbox_mask - true_object)
                tmp_neg_loc = torch.where(background_mask)

                # there is a chance that the object is small to not return a decent-sized bounding box
                # hence we might not find points sometimes there as well, hence we sample points from true background
                if torch.stack(tmp_neg_loc).shape[-1] == 0:
                    tmp_neg_loc = torch.where(true_object == 0)

            neg_index = np.random.choice(len(tmp_neg_loc[1]))
            neg_coordinates = [tmp_neg_loc[1][neg_index], tmp_neg_loc[2][neg_index]]
            neg_coordinates = neg_coordinates[::-1]
            neg_labels = 0

            negative_coordinates.append(neg_coordinates)
            negative_labels.append(neg_labels)

        return negative_coordinates, negative_labels

    def __call__(
        self,
        segmentation: torch.Tensor,
        prediction: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, None, None]:
        """Generate the prompts for each object iteratively in the segmentation.

        Args:
            The groundtruth segmentation. Expects a float tensor of shape NUM_OBJECTS x 1 x H x W.
            The predicted objects. Epects a float tensor of the same shape as the segmentation.

        Returns:
            The updated point prompt coordinates.
            The updated point prompt labels.
        """
        assert segmentation.shape == prediction.shape
        device = prediction.device

        true_object = segmentation.to(device)
        expected_diff = (prediction - true_object)
        neg_region = (expected_diff == 1).to(torch.float32)
        pos_region = (expected_diff == -1)
        overlap_region = torch.logical_and(prediction == 1, true_object == 1).to(torch.float32)

        pos_coordinates, pos_labels = self._get_positive_points(pos_region, overlap_region)
        neg_coordinates, neg_labels = self._get_negative_points(neg_region, true_object)
        assert len(pos_coordinates) == len(pos_labels) == len(neg_coordinates) == len(neg_labels)

        pos_coordinates = torch.tensor(pos_coordinates)[:, None]
        neg_coordinates = torch.tensor(neg_coordinates)[:, None]
        pos_labels, neg_labels = torch.tensor(pos_labels)[:, None], torch.tensor(neg_labels)[:, None]

        net_coords = torch.cat([pos_coordinates, neg_coordinates], dim=1)
        net_labels = torch.cat([pos_labels, neg_labels], dim=1)

        return net_coords, net_labels, None, None


"""
Helper functions for downloading Segment Anything models and predicting image embeddings.
"""

import hashlib
import os
import pickle
import warnings
from collections import OrderedDict
from shutil import copyfileobj
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import imageio.v3 as imageio
import numpy as np
import pooch
import requests
import torch
import vigra
import zarr
from elf.io import open_file
from nifty.tools import blocking
from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential

try:
    from mobile_sam import SamPredictor, sam_model_registry
    VIT_T_SUPPORT = True
except ImportError:
    from segment_anything import SamPredictor, sam_model_registry
    VIT_T_SUPPORT = False

try:
    from napari.utils import progress as tqdm
except ImportError:
    from tqdm import tqdm

_MODEL_URLS = {
    # the default segment anything models
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    # the model with vit tiny backend fom https://github.com/ChaoningZhang/MobileSAM
    "vit_t": "https://owncloud.gwdg.de/index.php/s/TuDzuwVDHd1ZDnQ/download",
    # first version of finetuned models on zenodo
    "vit_h_lm": "https://zenodo.org/record/8250299/files/vit_h_lm.pth?download=1",
    "vit_b_lm": "https://zenodo.org/record/8250281/files/vit_b_lm.pth?download=1",
    "vit_h_em": "https://zenodo.org/record/8250291/files/vit_h_em.pth?download=1",
    "vit_b_em": "https://zenodo.org/record/8250260/files/vit_b_em.pth?download=1",
}
_CACHE_DIR = os.environ.get('MICROSAM_CACHEDIR') or pooch.os_cache('micro_sam')
_CHECKPOINT_FOLDER = os.path.join(_CACHE_DIR, 'models')
_CHECKSUMS = {
    # the default segment anything models
    "vit_h": "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e",
    "vit_l": "3adcc4315b642a4d2101128f611684e8734c41232a17c648ed1693702a49a622",
    "vit_b": "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
    # the model with vit tiny backend fom https://github.com/ChaoningZhang/MobileSAM
    "vit_t": "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f",
    # first version of finetuned models on zenodo
    "vit_h_lm": "9a65ee0cddc05a98d60469a12a058859c89dc3ea3ba39fed9b90d786253fbf26",
    "vit_b_lm": "5a59cc4064092d54cd4d92cd967e39168f3760905431e868e474d60fe5464ecd",
    "vit_h_em": "ae3798a0646c8df1d4db147998a2d37e402ff57d3aa4e571792fbb911d8a979c",
    "vit_b_em": "c04a714a4e14a110f0eec055a65f7409d54e6bf733164d2933a0ce556f7d6f81",
}
# this is required so that the downloaded file is not called 'download'
_DOWNLOAD_NAMES = {
    "vit_t": "vit_t_mobile_sam.pth",
    "vit_h_lm": "vit_h_lm.pth",
    "vit_b_lm": "vit_b_lm.pth",
    "vit_h_em": "vit_h_em.pth",
    "vit_b_em": "vit_b_em.pth",
}
# this is the default model used in micro_sam
# currently set to the default vit_h
_DEFAULT_MODEL = "vit_h"


# TODO define the proper type for image embeddings
ImageEmbeddings = Dict[str, Any]
"""@private"""


#
# Functionality for model download and export
#


def _download(url, path, model_type):
    with requests.get(url, stream=True, verify=True) as r:
        if r.status_code != 200:
            r.raise_for_status()
            raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
        file_size = int(r.headers.get("Content-Length", 0))
        desc = f"Download {url} to {path}"
        if file_size == 0:
            desc += " (unknown file size)"
        with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw, open(path, "wb") as f:
            copyfileobj(r_raw, f)

    # validate the checksum
    expected_checksum = _CHECKSUMS[model_type]
    if expected_checksum is None:
        return
    with open(path, "rb") as f:
        file_ = f.read()
        checksum = hashlib.sha256(file_).hexdigest()
    if checksum != expected_checksum:
        raise RuntimeError(
            "The checksum of the download does not match the expected checksum."
            f"Expected: {expected_checksum}, got: {checksum}"
        )
    print("Download successful and checksums agree.")


def _get_checkpoint(model_type, checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_url = _MODEL_URLS[model_type]
        checkpoint_name = _DOWNLOAD_NAMES.get(model_type, checkpoint_url.split("/")[-1])
        checkpoint_path = os.path.join(_CHECKPOINT_FOLDER, checkpoint_name)

        # download the checkpoint if necessary
        if not os.path.exists(checkpoint_path):
            os.makedirs(_CHECKPOINT_FOLDER, exist_ok=True)
            _download(checkpoint_url, checkpoint_path, model_type)
    elif not os.path.exists(checkpoint_path):
        raise ValueError(f"The checkpoint path {checkpoint_path} that was passed does not exist.")

    return checkpoint_path


def _get_default_device():
    # Use cuda enabled gpu if it's available.
    if torch.cuda.is_available():
        device = "cuda"
    # As second priority use mps.
    # See https://pytorch.org/docs/stable/notes/mps.html for details
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("Using apple MPS device.")
        device = "mps"
    # Use the CPU as fallback.
    else:
        device = "cpu"
    return device


def _get_device(device=None):
    if device is None or device == "auto":
        device = _get_default_device()
    else:
        if device.lower() == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("PyTorch CUDA backend is not available.")
        elif device.lower() == "mps":
            if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
                raise RuntimeError("PyTorch MPS backend is not available or is not built correctly.")
        elif device.lower() == "cpu":
            pass  # cpu is always available
        else:
            raise RuntimeError(f"Unsupported device: {device}\n"
                               "Please choose from 'cpu', 'cuda', or 'mps'.")
    return device


def _available_devices():
    available_devices = []
    for i in ["cuda", "mps", "cpu"]:
        try:
            device = _get_device(i)
        except RuntimeError:
            pass
        else:
            available_devices.append(device)
    return available_devices


def get_sam_model(
    model_type: str = _DEFAULT_MODEL,
    device: Optional[str] = None,
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    return_sam: bool = False,
) -> SamPredictor:
    r"""Get the SegmentAnything Predictor.

    This function will download the required model checkpoint or load it from file if it
    was already downloaded.
    This location can be changed by setting the environment variable: MICROSAM_CACHEDIR.

    By default the models are downloaded to a folder named 'micro_sam/models'
    inside your default cache directory, eg:
    * Mac: ~/Library/Caches/<AppName>
    * Unix: ~/.cache/<AppName> or the value of the XDG_CACHE_HOME environment variable, if defined.
    * Windows: C:\Users\<user>\AppData\Local\<AppAuthor>\<AppName>\Cache
    See the pooch.os_cache() documentation for more details:
    https://www.fatiando.org/pooch/latest/api/generated/pooch.os_cache.html

    Args:
        device: The device for the model. If none is given will use GPU if available.
        model_type: The SegmentAnything model to use. Will use the standard vit_h model by default.
        checkpoint_path: The path to the corresponding checkpoint if not in the default model folder.
        return_sam: Return the sam model object as well as the predictor.

    Returns:
        The segment anything predictor.
    """
    checkpoint = _get_checkpoint(model_type, checkpoint_path)
    device = _get_device(device)

    # Our custom model types have a suffix "_...". This suffix needs to be stripped
    # before calling sam_model_registry.
    model_type_ = model_type[:5]
    assert model_type_ in ("vit_h", "vit_b", "vit_l", "vit_t")
    if model_type == "vit_t" and not VIT_T_SUPPORT:
        raise RuntimeError(
            "mobile_sam is required for the vit-tiny."
            "You can install it via 'pip install git+https://github.com/ChaoningZhang/MobileSAM.git'"
        )

    sam = sam_model_registry[model_type_](checkpoint=checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.model_type = model_type
    if return_sam:
        return predictor, sam
    return predictor


# We write a custom unpickler that skips objects that cannot be found instead of
# throwing an AttributeError or ModueNotFoundError.
# NOTE: since we just want to unpickle the model to load its weights these errors don't matter.
# See also https://stackoverflow.com/questions/27732354/unable-to-load-files-using-pickle-and-multiple-modules
class _CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError) as e:
            warnings.warn(f"Did not find {module}:{name} and will skip it, due to error {e}")
            return None


def get_custom_sam_model(
    checkpoint_path: Union[str, os.PathLike],
    model_type: str = "vit_h",
    device: Optional[str] = None,
    return_sam: bool = False,
    return_state: bool = False,
) -> SamPredictor:
    """Load a SAM model from a torch_em checkpoint.

    This function enables loading from the checkpoints saved by
    the functionality in `micro_sam.training`.

    Args:
        checkpoint_path: The path to the corresponding checkpoint if not in the default model folder.
        device: The device for the model. If none is given will use GPU if available.
        model_type: The SegmentAnything model to use.
        return_sam: Return the sam model object as well as the predictor.
        return_state: Return the full state of the checkpoint in addition to the predictor.

    Returns:
        The segment anything predictor.
    """
    assert not (return_sam and return_state)

    # over-ride the unpickler with our custom one
    custom_pickle = pickle
    custom_pickle.Unpickler = _CustomUnpickler

    device = _get_device(device)
    sam = sam_model_registry[model_type]()

    # load the model state, ignoring any attributes that can't be found by pickle
    state = torch.load(checkpoint_path, map_location=device, pickle_module=custom_pickle)
    model_state = state["model_state"]

    # copy the model weights from torch_em's training format
    sam_prefix = "sam."
    model_state = OrderedDict(
        [(k[len(sam_prefix):] if k.startswith(sam_prefix) else k, v) for k, v in model_state.items()]
    )
    sam.load_state_dict(model_state)
    sam.to(device)

    predictor = SamPredictor(sam)
    predictor.model_type = model_type

    if return_sam:
        return predictor, sam
    if return_state:
        return predictor, state
    return predictor


def export_custom_sam_model(
    checkpoint_path: Union[str, os.PathLike],
    model_type: str,
    save_path: Union[str, os.PathLike],
) -> None:
    """Export a finetuned segment anything model to the standard model format.

    The exported model can be used by the interactive annotation tools in `micro_sam.annotator`.

    Args:
        checkpoint_path: The path to the corresponding checkpoint if not in the default model folder.
        model_type: The SegmentAnything model type to use (vit_h, vit_b or vit_l).
        save_path: Where to save the exported model.
    """
    _, state = get_custom_sam_model(
        checkpoint_path, model_type=model_type, return_state=True, device="cpu",
    )
    model_state = state["model_state"]
    prefix = "sam."
    model_state = OrderedDict(
        [(k[len(prefix):] if k.startswith(prefix) else k, v) for k, v in model_state.items()]
    )
    torch.save(model_state, save_path)


def get_model_names() -> Iterable:
    return _MODEL_URLS.keys()


#
# Functionality for precomputing embeddings and other state
#


def _to_image(input_):
    # we require the input to be uint8
    if input_.dtype != np.dtype("uint8"):
        # first normalize the input to [0, 1]
        input_ = input_.astype("float32") - input_.min()
        input_ = input_ / input_.max()
        # then bring to [0, 255] and cast to uint8
        input_ = (input_ * 255).astype("uint8")
    if input_.ndim == 2:
        image = np.concatenate([input_[..., None]] * 3, axis=-1)
    elif input_.ndim == 3 and input_.shape[-1] == 3:
        image = input_
    else:
        raise ValueError(f"Invalid input image of shape {input_.shape}. Expect either 2D grayscale or 3D RGB image.")
    return image


def _precompute_tiled_2d(predictor, input_, tile_shape, halo, f, verbose=True):
    tiling = blocking([0, 0], input_.shape[:2], tile_shape)
    n_tiles = tiling.numberOfBlocks

    f.attrs["input_size"] = None
    f.attrs["original_size"] = None

    features = f.require_group("features")
    features.attrs["shape"] = input_.shape[:2]
    features.attrs["tile_shape"] = tile_shape
    features.attrs["halo"] = halo

    for tile_id in tqdm(range(n_tiles), total=n_tiles, desc="Predict image embeddings for tiles", disable=not verbose):
        tile = tiling.getBlockWithHalo(tile_id, list(halo))
        outer_tile = tuple(slice(beg, end) for beg, end in zip(tile.outerBlock.begin, tile.outerBlock.end))

        predictor.reset_image()
        tile_input = _to_image(input_[outer_tile])
        predictor.set_image(tile_input)
        tile_features = predictor.get_image_embedding()
        original_size = predictor.original_size
        input_size = predictor.input_size

        ds = features.create_dataset(
            str(tile_id), data=tile_features.cpu().numpy(), compression="gzip", chunks=tile_features.shape
        )
        ds.attrs["original_size"] = original_size
        ds.attrs["input_size"] = input_size

    return features


def _precompute_tiled_3d(predictor, input_, tile_shape, halo, f, verbose=True):
    assert input_.ndim == 3

    shape = input_.shape[1:]
    tiling = blocking([0, 0], shape, tile_shape)
    n_tiles = tiling.numberOfBlocks

    f.attrs["input_size"] = None
    f.attrs["original_size"] = None

    features = f.require_group("features")
    features.attrs["shape"] = shape
    features.attrs["tile_shape"] = tile_shape
    features.attrs["halo"] = halo

    n_slices = input_.shape[0]
    pbar = tqdm(total=n_tiles * n_slices, desc="Predict image embeddings for tiles and slices", disable=not verbose)

    for tile_id in range(n_tiles):
        tile = tiling.getBlockWithHalo(tile_id, list(halo))
        outer_tile = tuple(slice(beg, end) for beg, end in zip(tile.outerBlock.begin, tile.outerBlock.end))

        ds = None
        for z in range(n_slices):
            predictor.reset_image()
            tile_input = _to_image(input_[z][outer_tile])
            predictor.set_image(tile_input)
            tile_features = predictor.get_image_embedding()

            if ds is None:
                shape = (input_.shape[0],) + tile_features.shape
                chunks = (1,) + tile_features.shape
                ds = features.create_dataset(
                    str(tile_id), shape=shape, dtype="float32", compression="gzip", chunks=chunks
                )

            ds[z] = tile_features.cpu().numpy()
            pbar.update(1)

        original_size = predictor.original_size
        input_size = predictor.input_size

        ds.attrs["original_size"] = original_size
        ds.attrs["input_size"] = input_size

    return features


def _compute_2d(input_, predictor):
    image = _to_image(input_)
    predictor.set_image(image)
    features = predictor.get_image_embedding()
    original_size = predictor.original_size
    input_size = predictor.input_size
    image_embeddings = {
        "features": features.cpu().numpy(), "input_size": input_size, "original_size": original_size,
    }
    return image_embeddings


def _precompute_2d(input_, predictor, save_path, tile_shape, halo):
    f = zarr.open(save_path, "a")

    use_tiled_prediction = tile_shape is not None
    if "input_size" in f.attrs:  # the embeddings have already been precomputed
        features = f["features"][:] if tile_shape is None else f["features"]
        original_size, input_size = f.attrs["original_size"], f.attrs["input_size"]

    elif use_tiled_prediction:  # the embeddings have not been computed yet and we use tiled prediction
        features = _precompute_tiled_2d(predictor, input_, tile_shape, halo, f)
        original_size, input_size = None, None

    else:  # the embeddings have not been computed yet and we use normal prediction
        image = _to_image(input_)
        predictor.set_image(image)
        features = predictor.get_image_embedding()
        original_size, input_size = predictor.original_size, predictor.input_size
        f.create_dataset("features", data=features.cpu().numpy(), chunks=features.shape)
        f.attrs["input_size"] = input_size
        f.attrs["original_size"] = original_size

    image_embeddings = {
        "features": features, "input_size": input_size, "original_size": original_size,
    }
    return image_embeddings


def _compute_3d(input_, predictor):
    features = []
    original_size, input_size = None, None

    for z_slice in tqdm(input_, desc="Precompute Image Embeddings"):
        predictor.reset_image()

        image = _to_image(z_slice)
        predictor.set_image(image)
        embedding = predictor.get_image_embedding()
        features.append(embedding[None])

        if original_size is None:
            original_size = predictor.original_size
        if input_size is None:
            input_size = predictor.input_size

    # concatenate across the z axis
    features = torch.cat(features)

    image_embeddings = {
        "features": features.cpu().numpy(), "input_size": input_size, "original_size": original_size,
    }
    return image_embeddings


def _precompute_3d(input_, predictor, save_path, lazy_loading, tile_shape=None, halo=None):
    f = zarr.open(save_path, "a")

    use_tiled_prediction = tile_shape is not None
    if "input_size" in f.attrs:  # the embeddings have already been precomputed
        features = f["features"]
        original_size, input_size = f.attrs["original_size"], f.attrs["input_size"]

    elif use_tiled_prediction:  # the embeddings have not been computed yet and we use tiled prediction
        features = _precompute_tiled_3d(predictor, input_, tile_shape, halo, f)
        original_size, input_size = None, None

    else:  # the embeddings have not been computed yet and we use normal prediction
        features = f["features"] if "features" in f else None
        original_size, input_size = None, None

        for z, z_slice in tqdm(enumerate(input_), total=input_.shape[0], desc="Precompute Image Embeddings"):
            if features is not None:
                emb = features[z]
                if np.count_nonzero(emb) != 0:
                    continue

            predictor.reset_image()
            image = _to_image(z_slice)
            predictor.set_image(image)
            embedding = predictor.get_image_embedding()

            original_size, input_size = predictor.original_size, predictor.input_size
            if features is None:
                shape = (input_.shape[0],) + embedding.shape
                chunks = (1,) + embedding.shape
                features = f.create_dataset("features", shape=shape, chunks=chunks, dtype="float32")
            features[z] = embedding.cpu().numpy()

        f.attrs["input_size"] = input_size
        f.attrs["original_size"] = original_size

    # we load the data into memory if lazy loading was not specified
    # and if we do not use tiled prediction (we cannot load the full tiled data structure into memory)
    if not lazy_loading and not use_tiled_prediction:
        features = features[:]

    image_embeddings = {
        "features": features, "input_size": input_size, "original_size": original_size,
    }
    return image_embeddings


def _compute_data_signature(input_):
    data_signature = hashlib.sha1(np.asarray(input_).tobytes()).hexdigest()
    return data_signature


def precompute_image_embeddings(
    predictor: SamPredictor,
    input_: np.ndarray,
    save_path: Optional[str] = None,
    lazy_loading: bool = False,
    ndim: Optional[int] = None,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    wrong_file_callback: Optional[Callable] = None,
) -> ImageEmbeddings:
    """Compute the image embeddings (output of the encoder) for the input.

    If 'save_path' is given the embeddings will be loaded/saved in a zarr container.

    Args:
        predictor: The SegmentAnything predictor.
        input_: The input data. Can be 2 or 3 dimensional, corresponding to an image, volume or timeseries.
        save_path: Path to save the embeddings in a zarr container.
        lazy_loading: Whether to load all embeddings into memory or return an
            object to load them on demand when required. This only has an effect if 'save_path' is given
            and if the input is 3 dimensional.
        ndim: The dimensionality of the data. If not given will be deduced from the input data.
        tile_shape: Shape of tiles for tiled prediction. By default prediction is run without tiling.
        halo: Overlap of the tiles for tiled prediction.
        wrong_file_callback [callable]: Function to call when an embedding file with wrong file signature
            is passed. If none is given a wrong file signature will cause a warning.
            The callback ,ust have the signature 'def callback(save_path: str) -> str',
            where the return value is the (potentially updated) embedding save path.
    """
    ndim = input_.ndim if ndim is None else ndim
    if tile_shape is not None:
        assert save_path is not None, "Tiled prediction is only supported when the embeddings are saved to file."

    if save_path is not None:
        data_signature = _compute_data_signature(input_)

        f = zarr.open(save_path, "a")
        key_vals = [
            ("data_signature", data_signature),
            ("tile_shape", tile_shape if tile_shape is None else list(tile_shape)),
            ("halo", halo if halo is None else list(halo)),
            ("model_type", predictor.model_type)
        ]
        if "input_size" in f.attrs:  # we have computed the embeddings already and perform checks
            for key, val in key_vals:
                if val is None:
                    continue
                # check whether the key signature does not match or is not in the file
                if key not in f.attrs or f.attrs[key] != val:
                    raise RuntimeError(
                        f"Embeddings file {save_path} is invalid due to unmatching {key}: "
                        f"{f.attrs.get(key)} != {val}.Please recompute embeddings in a new file."
                    )
                    if wrong_file_callback is not None:
                        save_path = wrong_file_callback(save_path)
                        f = zarr.open(save_path, "a")
                    break

        for key, val in key_vals:
            if key not in f.attrs:
                f.attrs[key] = val

    if ndim == 2:
        image_embeddings = _compute_2d(input_, predictor) if save_path is None else\
            _precompute_2d(input_, predictor, save_path, tile_shape, halo)

    elif ndim == 3:
        image_embeddings = _compute_3d(input_, predictor) if save_path is None else\
            _precompute_3d(input_, predictor, save_path, lazy_loading, tile_shape, halo)

    else:
        raise ValueError(f"Invalid dimesionality {input_.ndim}, expect 2 or 3 dim data.")

    return image_embeddings


def set_precomputed(
    predictor: SamPredictor,
    image_embeddings: ImageEmbeddings,
    i: Optional[int] = None
):
    """Set the precomputed image embeddings for a predictor.

    Arguments:
        predictor: The SegmentAnything predictor.
        image_embeddings: The precomputed image embeddings computed by `precompute_image_embeddings`.
        i: Index for the image data. Required if `image` has three spatial dimensions
            or a time dimension and two spatial dimensions.
    """
    device = predictor.device
    features = image_embeddings["features"]

    assert features.ndim in (4, 5)
    if features.ndim == 5 and i is None:
        raise ValueError("The data is 3D so an index i is needed.")
    elif features.ndim == 4 and i is not None:
        raise ValueError("The data is 2D so an index is not needed.")

    if i is None:
        predictor.features = features.to(device) if torch.is_tensor(features) else \
            torch.from_numpy(features[:]).to(device)
    else:
        predictor.features = features[i].to(device) if torch.is_tensor(features) else \
            torch.from_numpy(features[i]).to(device)
    predictor.original_size = image_embeddings["original_size"]
    predictor.input_size = image_embeddings["input_size"]
    predictor.is_image_set = True

    return predictor


#
# Misc functionality
#


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute the intersection over union of two masks.

    Args:
        mask1: The first mask.
        mask2: The second mask.

    Returns:
        The intersection over union of the two masks.
    """
    overlap = np.logical_and(mask1 == 1, mask2 == 1).sum()
    union = np.logical_or(mask1 == 1, mask2 == 1).sum()
    eps = 1e-7
    iou = float(overlap) / (float(union) + eps)
    return iou


def get_centers_and_bounding_boxes(
    segmentation: np.ndarray,
    mode: str = "v"
) -> Tuple[Dict[int, np.ndarray], Dict[int, tuple]]:
    """Returns the center coordinates of the foreground instances in the ground-truth.

    Args:
        segmentation: The segmentation.
        mode: Determines the functionality used for computing the centers.
        If 'v', the object's eccentricity centers computed by vigra are used.
        If 'p' the object's centroids computed by skimage are used.

    Returns:
        A dictionary that maps object ids to the corresponding centroid.
        A dictionary that maps object_ids to the corresponding bounding box.
    """
    assert mode in ["p", "v"], "Choose either 'p' for regionprops or 'v' for vigra"

    properties = regionprops(segmentation)

    if mode == "p":
        center_coordinates = {prop.label: prop.centroid for prop in properties}
    elif mode == "v":
        center_coordinates = vigra.filters.eccentricityCenters(segmentation.astype('float32'))
        center_coordinates = {i: coord for i, coord in enumerate(center_coordinates) if i > 0}

    bbox_coordinates = {prop.label: prop.bbox for prop in properties}

    assert len(bbox_coordinates) == len(center_coordinates), f"{len(bbox_coordinates)}, {len(center_coordinates)}"
    return center_coordinates, bbox_coordinates


def load_image_data(
    path: str,
    key: Optional[str] = None,
    lazy_loading: bool = False
) -> np.ndarray:
    """Helper function to load image data from file.

    Args:
        path: The filepath to the image data.
        key: The internal filepath for complex data formats like hdf5.
        lazy_loading: Whether to lazyly load data. Only supported for n5 and zarr data.

    Returns:
        The image data.
    """
    if key is None:
        image_data = imageio.imread(path)
    else:
        with open_file(path, mode="r") as f:
            image_data = f[key]
            if not lazy_loading:
                image_data = image_data[:]
    return image_data


def segmentation_to_one_hot(
    segmentation: np.ndarray,
    segmentation_ids: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Convert the segmentation to one-hot encoded masks.

    Args:
        segmentation: The segmentation.
        segmentation_ids: Optional subset of ids that will be used to subsample the masks.

    Returns:
        The one-hot encoded masks.
    """
    masks = segmentation.copy()
    if segmentation_ids is None:
        n_ids = int(segmentation.max())

    else:
        assert segmentation_ids[0] != 0

        # the segmentation ids have to be sorted
        segmentation_ids = np.sort(segmentation_ids)

        # set the non selected objects to zero and relabel sequentially
        masks[~np.isin(masks, segmentation_ids)] = 0
        masks = relabel_sequential(masks)[0]
        n_ids = len(segmentation_ids)

    masks = torch.from_numpy(masks)

    one_hot_shape = (n_ids + 1,) + masks.shape
    masks = masks.unsqueeze(0)  # add dimension to scatter
    masks = torch.zeros(one_hot_shape).scatter_(0, masks, 1)[1:]

    # add the extra singleton dimenion to get shape NUM_OBJECTS x 1 x H x W
    masks = masks.unsqueeze(1)
    return masks

# simple wrapper around SAM in order to keep things trainable
class TrainableSAM(nn.Module):
    """Wrapper to make the SegmentAnything model trainable.

    Args:
        sam: The SegmentAnything Model.
        device: The device for training.
    """
    def __init__(
        self,
        sam: Sam,
        device: Union[str, torch.device],
    ) -> None:
        super().__init__()
        self.sam = sam
        self.device = device

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input.

        Args:
            x: The input tensor.

        Returns:
            The normalized and padded tensor.
        """
        # Normalize colors
        x = (x - self.sam.pixel_mean) / self.sam.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.sam.image_encoder.img_size - h
        padw = self.sam.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def image_embeddings_oft(self, input_images):
        """@private"""
        image_embeddings = self.sam.image_encoder(input_images)
        return image_embeddings

    # batched inputs follow the same syntax as the input to sam.forward
    def forward(
        self,
        batched_inputs: List[Dict[str, Any]],
        multimask_output: bool = False,
        image_embeddings: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, Any]]:
        """Forward pass.

        Args:
            batched_inputs: The batched input images and prompts.
            multimask_output: Whether to predict mutiple or just a single mask.
            image_embeddings: The precompute image embeddings. If not passed then they will be computed.

        Returns:
            The predicted segmentation masks and iou values.
        """
        input_images = torch.stack([self.preprocess(x=x["image"].to(self.device)) for x in batched_inputs], dim=0)
        if image_embeddings is None:
            image_embeddings = self.sam.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_inputs, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"].to(self.device), image_record["point_labels"].to(self.device))
            else:
                points = None

            if "boxes" in image_record:
                boxes = image_record.get("boxes").to(self.device)
            else:
                boxes = None

            if "mask_inputs" in image_record:
                masks = image_record.get("mask_inputs").to(self.device)
            else:
                masks = None

            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=points,
                boxes=boxes,
                masks=masks,
            )

            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            masks = self.sam.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )

            outputs.append(
                {
                    "low_res_masks": low_res_masks,
                    "masks": masks,
                    "iou_predictions": iou_predictions
                }
            )

        return outputs

def get_trainable_sam_model(
    model_type: str = "vit_h",
    device: Optional[str] = None,
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    freeze: Optional[List[str]] = None,
) -> TrainableSAM:
    """Get the trainable sam model.

    Args:
        model_type: The type of the segment anything model.
        checkpoint_path: Path to a custom checkpoint from which to load the model weights.
        freeze: Specify parts of the model that should be frozen, namely: image_encoder, prompt_encoder and mask_decoder
            By default nothing is frozen and the full model is updated.
        device: The device to use for training.

    Returns:
        The trainable segment anything model.
    """
    # set the device here so that the correct one is passed to TrainableSAM below
    device = _get_device(device)
    _, sam = get_sam_model(model_type=model_type, device=device, checkpoint_path=checkpoint_path, return_sam=True)

    # freeze components of the model if freeze was passed
    # ideally we would want to add components in such a way that:
    # - we would be able to freeze the choice of encoder/decoder blocks, yet be able to add components to the network
    #   (for e.g. encoder blocks to "image_encoder")
    if freeze is not None:
        for name, param in sam.named_parameters():
            if isinstance(freeze, list):
                # we would want to "freeze" all the components in the model if passed a list of parts
                for l_item in freeze:
                    if name.startswith(f"{l_item}"):
                        param.requires_grad = False
            else:
                # we "freeze" only for one specific component when passed a "particular" part
                if name.startswith(f"{freeze}"):
                    param.requires_grad = False

    # convert to trainable sam
    trainable_sam = TrainableSAM(sam, device)
    return trainable_sam


class ConvertToSamInputs:
    """Convert outputs of data loader to the expected batched inputs of the SegmentAnything model.

    Args:
        dilation_strength: The dilation factor.
            It determines a "safety" border from which prompts are not sampled to avoid ambiguous prompts
            due to imprecise groundtruth masks.
        box_distortion_factor: Factor for distorting the box annotations derived from the groundtruth masks.
            Not yet implemented.
    """
    def __init__(
        self,
        dilation_strength: int = 10,
        box_distortion_factor: Optional[float] = None,
    ) -> None:
        self.dilation_strength = dilation_strength
        # TODO implement the box distortion logic
        if box_distortion_factor is not None:
            raise NotImplementedError

    def _get_prompt_lists(self, gt, n_samples, prompt_generator):
        """Returns a list of "expected" prompts subjected to the random input attributes for prompting."""

        _, bbox_coordinates = get_centers_and_bounding_boxes(gt, mode="p")

        # get the segment ids
        cell_ids = np.unique(gt)[1:]
        if n_samples is None:  # n-samples is set to None, so we use all ids
            sampled_cell_ids = cell_ids

        else:  # n-samples is set, so we subsample the cell ids
            sampled_cell_ids = np.random.choice(cell_ids, size=min(n_samples, len(cell_ids)), replace=False)
            sampled_cell_ids = np.sort(sampled_cell_ids)

        # only keep the bounding boxes for sampled cell ids
        bbox_coordinates = [bbox_coordinates[sampled_id] for sampled_id in sampled_cell_ids]

        # convert the gt to the one-hot-encoded masks for the sampled cell ids
        object_masks = segmentation_to_one_hot(gt, None if n_samples is None else sampled_cell_ids)

        # derive and return the prompts
        point_prompts, point_label_prompts, box_prompts, _ = prompt_generator(object_masks, bbox_coordinates)
        return box_prompts, point_prompts, point_label_prompts, sampled_cell_ids

    def __call__(self, x, y, n_pos, n_neg, get_boxes=False, n_samples=None):
        """Convert the outputs of dataloader and prompt settings to the batch format expected by SAM.
        """

        # condition to see if we get point prompts, then we (ofc) use point-prompting
        # else we don't use point prompting
        if n_pos == 0 and n_neg == 0:
            get_points = False
        else:
            get_points = True

        # keeping the solution open by checking for deterministic/dynamic choice of point prompts
        prompt_generator = PointAndBoxPromptGenerator(n_positive_points=n_pos,
                                                      n_negative_points=n_neg,
                                                      dilation_strength=self.dilation_strength,
                                                      get_box_prompts=get_boxes,
                                                      get_point_prompts=get_points)

        batched_inputs = []
        batched_sampled_cell_ids_list = []

        for image, gt in zip(x, y):
            gt = gt.squeeze().numpy().astype(np.int64)
            box_prompts, point_prompts, point_label_prompts, sampled_cell_ids = self._get_prompt_lists(
                gt, n_samples, prompt_generator,
            )

            # check to be sure about the expected size of the no. of elements in different settings
            if get_boxes:
                assert len(sampled_cell_ids) == len(box_prompts), f"{len(sampled_cell_ids)}, {len(box_prompts)}"

            if get_points:
                assert len(sampled_cell_ids) == len(point_prompts) == len(point_label_prompts), \
                    f"{len(sampled_cell_ids)}, {len(point_prompts)}, {len(point_label_prompts)}"

            batched_sampled_cell_ids_list.append(sampled_cell_ids)

            batched_input = {"image": image, "original_size": image.shape[1:]}
            if get_boxes:
                batched_input["boxes"] = box_prompts
            if get_points:
                batched_input["point_coords"] = point_prompts
                batched_input["point_labels"] = point_label_prompts

            batched_inputs.append(batched_input)

        return batched_inputs, batched_sampled_cell_ids_list


class SamTrainer(torch_em.trainer.DefaultTrainer):
    """Trainer class for training the Segment Anything model.

    This class is derived from `torch_em.trainer.DefaultTrainer`.
    Check out https://github.com/constantinpape/torch-em/blob/main/torch_em/trainer/default_trainer.py
    for details on its usage and implementation.

    Args:
        convert_inputs: The class that converts outputs of the dataloader to the expected input format of SAM.
            The class `micro_sam.training.util.ConvertToSamInputs` can be used here.
        n_sub_iteration: The number of iteration steps for which the masks predicted for one object are updated.
            In each sub-iteration new point prompts are sampled where the model was wrong.
        n_objects_per_batch: If not given, we compute the loss for all objects in a sample.
            Otherwise the loss computation is limited to n_objects_per_batch, and the objects are randomly sampled.
        mse_loss: The regression loss to compare the IoU predicted by the model with the true IoU.
        sigmoid: The activation function for normalizing the model output.
        prompt_generator: The iterative prompt generator which takes care of the iterative prompting logic for training
        mask_prob: The probability of using the mask inputs in the iterative prompting (per `n_sub_iteration`)
        **kwargs: The keyword arguments of the DefaultTrainer super class.
    """

    def __init__(
        self,
        convert_inputs,
        n_sub_iteration: int,
        n_objects_per_batch: Optional[int] = None,
        mse_loss: torch.nn.Module = torch.nn.MSELoss(),
        _sigmoid: torch.nn.Module = torch.nn.Sigmoid(),
        prompt_generator: PromptGeneratorBase = IterativePromptGenerator(),
        mask_prob: float = 0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.convert_inputs = convert_inputs
        self.mse_loss = mse_loss
        self._sigmoid = _sigmoid
        self.n_objects_per_batch = n_objects_per_batch
        self.n_sub_iteration = n_sub_iteration
        self.prompt_generator = prompt_generator
        self.mask_prob = mask_prob
        self._kwargs = kwargs

    def _get_prompt_and_multimasking_choices(self, current_iteration):
        """Choose the type of prompts we sample for training, and then we call
        'convert_inputs' with the correct prompting from here.
        """
        if current_iteration % 2 == 0:  # sample only a single point per object
            n_pos, n_neg = 1, 0
            get_boxes = False
            multimask_output = True

        else:  # sample only a single box per object
            n_pos, n_neg = 0, 0
            get_boxes = True
            multimask_output = False

        return n_pos, n_neg, get_boxes, multimask_output

    def _get_prompt_and_multimasking_choices_for_val(self, current_iteration):
        """Choose the type of prompts we sample for validation, and then we call
        'convert_inputs' with the correct prompting from here.
        """
        if current_iteration % 4 == 0:  # sample only a single point per object
            n_pos, n_neg = 1, 0
            get_boxes = False
            multimask_output = True

        elif current_iteration % 4 == 1:  # sample only a single box per object
            n_pos, n_neg = 0, 0
            get_boxes = True
            multimask_output = False

        elif current_iteration % 4 == 2:  # sample a random no. of points
            pos_range, neg_range = 4, 4

            n_pos = np.random.randint(1, pos_range + 1)
            if n_pos == 1:  # to avoid (1, 0) combination for redundancy but still have (n_pos, 0)
                n_neg = np.random.randint(1, neg_range + 1)
            else:
                n_neg = np.random.randint(0, neg_range + 1)
            get_boxes = False
            multimask_output = False

        else:  # sample boxes AND random no. of points
            # here we can have (1, 0) because we also have box
            pos_range, neg_range = 4, 4

            n_pos = np.random.randint(1, pos_range + 1)
            n_neg = np.random.randint(0, neg_range + 1)
            get_boxes = True
            multimask_output = False

        return n_pos, n_neg, get_boxes, multimask_output

    def _get_dice(self, input_, target):
        """Using the default "DiceLoss" called by the trainer from "torch_em"
        """
        dice_loss = self.loss(input_, target)
        return dice_loss

    def _get_iou(self, pred, true, eps=1e-7):
        """Getting the IoU score for the predicted and true labels
        """
        pred_mask = pred > 0.5  # binarizing the output predictions
        overlap = pred_mask.logical_and(true).sum()
        union = pred_mask.logical_or(true).sum()
        iou = overlap / (union + eps)
        return iou

    def _get_net_loss(self, batched_outputs, y, sampled_ids):
        """What do we do here? two **separate** things
        1. compute the mask loss: loss between the predicted and ground-truth masks
            for this we just use the dice of the prediction vs. the gt (binary) mask
        2. compute the mask for the "IOU Regression Head": so we want the iou output from the decoder to
            match the actual IOU between predicted and (binary) ground-truth mask. And we use L2Loss / MSE for this.
        """
        masks = [m["masks"] for m in batched_outputs]
        predicted_iou_values = [m["iou_predictions"] for m in batched_outputs]
        with torch.no_grad():
            mean_model_iou = torch.mean(torch.stack([p.mean() for p in predicted_iou_values]))

        mask_loss = 0.0  # this is the loss term for 1.
        iou_regression_loss = 0.0  # this is the loss term for 2.

        # outer loop is over the batch (different image/patch predictions)
        for m_, y_, ids_, predicted_iou_ in zip(masks, y, sampled_ids, predicted_iou_values):
            per_object_dice_scores, per_object_iou_scores = [], []

            # inner loop is over the channels, this corresponds to the different predicted objects
            for i, (predicted_obj, predicted_iou) in enumerate(zip(m_, predicted_iou_)):
                predicted_obj = self._sigmoid(predicted_obj).to(self.device)
                true_obj = (y_ == ids_[i]).to(self.device)

                # this is computing the LOSS for 1.)
                _dice_score = min([self._get_dice(p[None], true_obj) for p in predicted_obj])
                per_object_dice_scores.append(_dice_score)

                # now we need to compute the loss for 2.)
                with torch.no_grad():
                    true_iou = torch.stack([self._get_iou(p[None], true_obj) for p in predicted_obj])
                _iou_score = self.mse_loss(true_iou, predicted_iou)
                per_object_iou_scores.append(_iou_score)

            mask_loss = mask_loss + torch.mean(torch.stack(per_object_dice_scores))
            iou_regression_loss = iou_regression_loss + torch.mean(torch.stack(per_object_iou_scores))

        loss = mask_loss + iou_regression_loss

        return loss, mask_loss, iou_regression_loss, mean_model_iou

    def _postprocess_outputs(self, masks):
        """ "masks" look like -> (B, 1, X, Y)
        where, B is the number of objects, (X, Y) is the input image shape
        """
        instance_labels = []
        for m in masks:
            instance_list = [self._sigmoid(_val) for _val in m.squeeze(1)]
            instance_label = torch.stack(instance_list, dim=0).sum(dim=0).clip(0, 1)
            instance_labels.append(instance_label)
        instance_labels = torch.stack(instance_labels).unsqueeze(1)
        return instance_labels

    def _get_val_metric(self, batched_outputs, sampled_binary_y):
        """ Tracking the validation metric based on the DiceLoss
        """
        masks = [m["masks"] for m in batched_outputs]
        pred_labels = self._postprocess_outputs(masks)

        # we do the condition below to adapt w.r.t. the multimask output to select the "objectively" best response
        if pred_labels.dim() == 5:
            metric = min([self.metric(pred_labels[:, :, i, :, :], sampled_binary_y.to(self.device))
                          for i in range(pred_labels.shape[2])])
        else:
            metric = self.metric(pred_labels, sampled_binary_y.to(self.device))

        return metric

    #
    # Update Masks Iteratively while Training
    #
    def _update_masks(self, batched_inputs, y, sampled_binary_y, sampled_ids, num_subiter, multimask_output):
        # estimating the image inputs to make the computations faster for the decoder
        input_images = torch.stack([self.model.preprocess(x=x["image"].to(self.device)) for x in batched_inputs], dim=0)
        image_embeddings = self.model.image_embeddings_oft(input_images)

        loss, mask_loss, iou_regression_loss, mean_model_iou = 0.0, 0.0, 0.0, 0.0

        # this loop takes care of the idea of sub-iterations, i.e. the number of times we iterate over each batch
        for i in range(num_subiter):
            # we do multimasking only in the first sub-iteration as we then pass single prompt
            # after the first sub-iteration, we don't do multimasking because we get multiple prompts
            batched_outputs = self.model(batched_inputs,
                                         multimask_output=multimask_output if i == 0 else False,
                                         image_embeddings=image_embeddings)

            # we want to average the loss and then backprop over the net sub-iterations
            net_loss, net_mask_loss, net_iou_regression_loss, net_mean_model_iou = self._get_net_loss(batched_outputs,
                                                                                                      y, sampled_ids)
            loss += net_loss
            mask_loss += net_mask_loss
            iou_regression_loss += net_iou_regression_loss
            mean_model_iou += net_mean_model_iou

            masks, logits_masks = [], []
            # the loop below gets us the masks and logits from the batch-level outputs
            for m in batched_outputs:
                mask, l_mask = [], []
                for _m, _l, _iou in zip(m["masks"], m["low_res_masks"], m["iou_predictions"]):
                    best_iou_idx = torch.argmax(_iou)
                    best_mask, best_logits = _m[best_iou_idx][None], _l[best_iou_idx][None]
                    mask.append(self._sigmoid(best_mask))
                    l_mask.append(best_logits)

                mask, l_mask = torch.stack(mask), torch.stack(l_mask)
                masks.append(mask)
                logits_masks.append(l_mask)

            masks, logits_masks = torch.stack(masks), torch.stack(logits_masks)
            masks = (masks > 0.5).to(torch.float32)

            self._get_updated_points_per_mask_per_subiter(masks, sampled_binary_y, batched_inputs, logits_masks)

        loss = loss / num_subiter
        mask_loss = mask_loss / num_subiter
        iou_regression_loss = iou_regression_loss / num_subiter
        mean_model_iou = mean_model_iou / num_subiter

        return loss, mask_loss, iou_regression_loss, mean_model_iou

    def _get_updated_points_per_mask_per_subiter(self, masks, sampled_binary_y, batched_inputs, logits_masks):
        # here, we get the pair-per-batch of predicted and true elements (and also the "batched_inputs")
        for x1, x2, _inp, logits in zip(masks, sampled_binary_y, batched_inputs, logits_masks):
            # here, we get each object in the pairs and do the point choices per-object
            net_coords, net_labels, _, _ = self.prompt_generator(x2, x1)

            updated_point_coords = torch.cat([_inp["point_coords"], net_coords], dim=1) \
                if "point_coords" in _inp.keys() else net_coords
            updated_point_labels = torch.cat([_inp["point_labels"], net_labels], dim=1) \
                if "point_labels" in _inp.keys() else net_labels

            _inp["point_coords"] = updated_point_coords
            _inp["point_labels"] = updated_point_labels

            if self.mask_prob > 0:
                # using mask inputs for iterative prompting while training, with a probability
                use_mask_inputs = (random.random() < self.mask_prob)
                if use_mask_inputs:
                    _inp["mask_inputs"] = logits
                else:  # remove  previously existing mask inputs to avoid using them in next sub-iteration
                    _inp.pop("mask_inputs", None)

    #
    # Training Loop
    #

    def _update_samples_for_gt_instances(self, y, n_samples):
        num_instances_gt = torch.amax(y, dim=(1, 2, 3))
        num_instances_gt = num_instances_gt.numpy().astype(int)
        n_samples = min(num_instances_gt) if n_samples > min(num_instances_gt) else n_samples
        return n_samples

    def _train_epoch_impl(self, progress, forward_context, backprop):
        self.model.train()

        n_iter = 0
        t_per_iter = time.time()
        for x, y in self.train_loader:

            self.optimizer.zero_grad()

            with forward_context():
                n_samples = self._update_samples_for_gt_instances(y, self.n_objects_per_batch)

                n_pos, n_neg, get_boxes, multimask_output = self._get_prompt_and_multimasking_choices(self._iteration)

                batched_inputs, sampled_ids = self.convert_inputs(x, y, n_pos, n_neg, get_boxes, n_samples)

                assert len(y) == len(sampled_ids)
                sampled_binary_y = []
                for i in range(len(y)):
                    _sampled = [torch.isin(y[i], torch.tensor(idx)) for idx in sampled_ids[i]]
                    sampled_binary_y.append(_sampled)

                # the steps below are done for one reason in a gist:
                # to handle images where there aren't enough instances as expected
                # (e.g. where one image has only one instance)
                obj_lengths = [len(s) for s in sampled_binary_y]
                sampled_binary_y = [s[:min(obj_lengths)] for s in sampled_binary_y]
                sampled_binary_y = [torch.stack(s).to(torch.float32) for s in sampled_binary_y]
                sampled_binary_y = torch.stack(sampled_binary_y)

                # gist for below - while we find the mismatch, we need to update the batched inputs
                # else it would still generate masks using mismatching prompts, and it doesn't help us
                # with the subiterations again. hence we clip the number of input points as well
                f_objs = sampled_binary_y.shape[1]
                batched_inputs = [
                    {k: (v[:f_objs] if k in ("point_coords", "point_labels", "boxes") else v) for k, v in inp.items()}
                    for inp in batched_inputs
                ]

                loss, mask_loss, iou_regression_loss, model_iou = self._update_masks(batched_inputs, y,
                                                                                     sampled_binary_y, sampled_ids,
                                                                                     num_subiter=self.n_sub_iteration,
                                                                                     multimask_output=multimask_output)

            backprop(loss)

            if self.logger is not None:
                lr = [pm["lr"] for pm in self.optimizer.param_groups][0]
                samples = sampled_binary_y if self._iteration % self.log_image_interval == 0 else None
                self.logger.log_train(self._iteration, loss, lr, x, y, samples,
                                      mask_loss, iou_regression_loss, model_iou)

            self._iteration += 1
            n_iter += 1
            if self._iteration >= self.max_iteration:
                break
            progress.update(1)

        t_per_iter = (time.time() - t_per_iter) / n_iter
        return t_per_iter

    def _validate_impl(self, forward_context):
        self.model.eval()

        val_iteration = 0
        metric_val, loss_val, model_iou_val = 0.0, 0.0, 0.0

        with torch.no_grad():
            for x, y in self.val_loader:
                with forward_context():
                    n_samples = self._update_samples_for_gt_instances(y, self.n_objects_per_batch)

                    (n_pos, n_neg,
                     get_boxes, multimask_output) = self._get_prompt_and_multimasking_choices_for_val(val_iteration)

                    batched_inputs, sampled_ids = self.convert_inputs(x, y, n_pos, n_neg, get_boxes, n_samples)

                    batched_outputs = self.model(batched_inputs, multimask_output=multimask_output)

                    assert len(y) == len(sampled_ids)
                    sampled_binary_y = torch.stack(
                        [torch.isin(y[i], torch.tensor(sampled_ids[i])) for i in range(len(y))]
                    ).to(torch.float32)

                    loss, mask_loss, iou_regression_loss, model_iou = self._get_net_loss(batched_outputs,
                                                                                         y, sampled_ids)

                    metric = self._get_val_metric(batched_outputs, sampled_binary_y)

                loss_val += loss.item()
                metric_val += metric.item()
                model_iou_val += model_iou.item()
                val_iteration += 1

        loss_val /= len(self.val_loader)
        metric_val /= len(self.val_loader)
        model_iou_val /= len(self.val_loader)
        print()
        print(f"The Average Dice Score for the Current Epoch is {1 - metric_val}")

        if self.logger is not None:
            self.logger.log_validation(
                self._iteration, metric_val, loss_val, x, y,
                sampled_binary_y, mask_loss, iou_regression_loss, model_iou_val
            )

        return metric_val


class SamLogger(TorchEmLogger):
    """@private"""
    def __init__(self, trainer, save_root, **unused_kwargs):
        super().__init__(trainer, save_root)
        self.log_dir = f"./logs/{trainer.name}" if save_root is None else\
            os.path.join(save_root, "logs", trainer.name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.tb = torch.utils.tensorboard.SummaryWriter(self.log_dir)
        self.log_image_interval = trainer.log_image_interval

    def add_image(self, x, y, samples, name, step):
        self.tb.add_image(tag=f"{name}/input", img_tensor=x[0], global_step=step)
        self.tb.add_image(tag=f"{name}/target", img_tensor=y[0], global_step=step)
        sample_grid = make_grid([sample[0] for sample in samples], nrow=4, padding=4)
        self.tb.add_image(tag=f"{name}/samples", img_tensor=sample_grid, global_step=step)

    def log_train(self, step, loss, lr, x, y, samples, mask_loss, iou_regression_loss, model_iou):
        self.tb.add_scalar(tag="train/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="train/mask_loss", scalar_value=mask_loss, global_step=step)
        self.tb.add_scalar(tag="train/iou_loss", scalar_value=iou_regression_loss, global_step=step)
        self.tb.add_scalar(tag="train/model_iou", scalar_value=model_iou, global_step=step)
        self.tb.add_scalar(tag="train/learning_rate", scalar_value=lr, global_step=step)
        if step % self.log_image_interval == 0:
            self.add_image(x, y, samples, "train", step)

    def log_validation(self, step, metric, loss, x, y, samples, mask_loss, iou_regression_loss, model_iou):
        self.tb.add_scalar(tag="validation/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="validation/mask_loss", scalar_value=mask_loss, global_step=step)
        self.tb.add_scalar(tag="validation/iou_loss", scalar_value=iou_regression_loss, global_step=step)
        self.tb.add_scalar(tag="validation/model_iou", scalar_value=model_iou, global_step=step)
        self.tb.add_scalar(tag="validation/metric", scalar_value=metric, global_step=step)
        self.add_image(x, y, samples, "validation", step)





def get_dataloaders(patch_shape, data_path, cell_type=None):
    """This returns the livecell data loaders implemented in torch_em:
    https://github.com/constantinpape/torch-em/blob/main/torch_em/data/datasets/livecell.py
    It will automatically download the livecell data.

    Note: to replace this with another data loader you need to return a torch data loader
    that retuns `x, y` tensors, where `x` is the image data and `y` are the labels.
    The labels have to be in a label mask instance segmentation format.
    I.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID.
    Important: the ID 0 is reseved for background, and the IDs must be consecutive
    """
    label_transform = torch_em.transform.label.label_consecutive  # to ensure consecutive IDs
    train_loader = get_livecell_loader(path=data_path, patch_shape=patch_shape, split="train", batch_size=2,
                                       num_workers=16, cell_types=cell_type, download=True,
                                       label_transform=label_transform, shuffle=True)
    val_loader = get_livecell_loader(path=data_path, patch_shape=patch_shape, split="val", batch_size=1,
                                     num_workers=16, cell_types=cell_type, download=True,
                                     label_transform=label_transform, shuffle=True)
    return train_loader, val_loader


def finetune_livecell(args):
    """Example code for finetuning SAM on LiveCELL"""
    # override this (below) if you have some more complex set-up and need to specify the exact gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # training settings:
    model_type = args.model_type
    checkpoint_path = None  # override this to start training from a custom checkpoint
    patch_shape = (520, 704)  # the patch shape for training
    n_objects_per_batch = 25  # this is the number of objects per batch that will be sampled

    # get the trainable segment anything model
    model = get_trainable_sam_model(model_type=model_type, device=device, checkpoint_path=checkpoint_path)

    # all the stuff we need for training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=10, verbose=True)
    train_loader, val_loader = get_dataloaders(patch_shape=patch_shape, data_path=args.input_path)

    # this class creates all the training data for a batch (inputs, prompts and labels)
    convert_inputs = ConvertToSamInputs()

    checkpoint_name = "livecell_sam"
    # the trainer which performs training and validation (implemented using "torch_em")
    trainer = SamTrainer(
        name=checkpoint_name,
        save_root=args.save_root,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        # currently we compute loss batch-wise, else we pass channelwise True
        loss=torch_em.loss.DiceLoss(channelwise=False),
        metric=torch_em.loss.DiceLoss(),
        device=device,
        lr_scheduler=scheduler,
        logger=SamLogger,
        log_image_interval=100,
        mixed_precision=True,
        convert_inputs=convert_inputs,
        n_objects_per_batch=n_objects_per_batch,
        n_sub_iteration=8,
        compile_model=False,
        mask_prob=0.5  # (optional) overwrite to provide the probability of using mask inputs while training
    )
    trainer.fit(args.iterations)
    if args.export_path is not None:
        checkpoint_path = os.path.join(
            "" if args.save_root is None else args.save_root, "checkpoints", checkpoint_name, "best.pt"
        )
        export_custom_sam_model(
            checkpoint_path=checkpoint_path,
            model_type=model_type,
            save_path=args.export_path,
        )


def main():
    parser = argparse.ArgumentParser(description="Finetune Segment Anything for the LiveCELL dataset.")
    parser.add_argument(
        "--input_path", "-i", default="/scratch/projects/nim00007/data/LiveCELL/",
        help="The filepath to the LiveCELL data. If the data does not exist yet it will be downloaded."
    )
    parser.add_argument(
        "--model_type", "-m", default="vit_b",
        help="The model type to use for fine-tuning. Either vit_h, vit_b or vit_l."
    )
    parser.add_argument(
        "--save_root", "-s",
        help="Where to save the checkpoint and logs. By default they will be saved where this script is run."
    )
    parser.add_argument(
        "--iterations", type=int, default=int(1e5),
        help="For how many iterations should the model be trained? By default 100k."
    )
    parser.add_argument(
        "--export_path", "-e",
        help="Where to export the finetuned model to. The exported model can be used in the annotation tools."
    )
    args = parser.parse_args()
    finetune_livecell(args)


if __name__ == "__main__":
    main()
