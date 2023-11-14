import numpy as np
import torch.nn as nn
import torch_em
import torch_em.data.datasets as torchem_data
from torch_em.model import UNet2d
from torch_em.util.debug import check_loader, check_trainer


# To train the generalist model using with LM,
# They use 6 datasets, named as below!

dataset_names = [
    #"covid_if", 
    #"dsb", 
    #"hpa", 
    #"isbi2012", 
    "livecell", 
    #"vnc-mitos",
    #"DeepBacs", 
    #"Lizard", 
    #"Mouse_Embryo", 
    #"NeuIPS22-CellSeg",
    #"Nucleus_dsb", 
    #"PlantSeg", 
    #"TissueNet"
]


# Use a pre-configured dataset
# Specify a pre-configured dataset. Set to `None` in order to specify the training data via file-paths instead.
preconfigured_dataset = None

# Where to download the training data (the data will be downloaded only once).
# If you work in google colab you may want to adapt this path to be on your google drive, in order
# to not loose the data after each session.
#download_folder = f'./database/LM/{preconfigured_dataset}'

download_folder = f'/proj/aicell/users/linh/cv/micro-sam-ori/database/LM/livecell'

# Create a custom dataset from local data by specifiying the paths for training data, training labels
# as well as validation data and validation labels
train_data_paths = []
val_data_paths = []
data_key = ""
train_label_paths = []
val_label_paths = []
label_key = ""

# In addition you can also specify region of interests for training using the normal python slice syntax
train_rois = None
val_rois = None

# This should be chosen s.t. it is smaller than the smallest image in your training data.
# If you are training from 3d data (data with a z-axis), you will need to specify the patch_shape
# as (1, shape_y, shape_x).
patch_shape = (512, 512)


data_paths = f'/proj/aicell/users/linh/cv/micro-sam-ori/database/LM/livecell/'

def check_data(data_paths, label_paths, rois):
    print("Loading the raw data from:", data_paths, data_key)
    print("Loading the labels from:", label_paths, label_key)
    try:
        torch_em.default_segmentation_dataset(data_paths, data_key, label_paths, label_key, patch_shape, rois=rois)
    except Exception as e:
        print("Loading the dataset failed with:")
        raise e

if preconfigured_dataset is None:
    print("Using a custom dataset:")
    print("Checking the training dataset:")
    check_data(train_data_paths, train_label_paths, train_rois)
    check_data(val_data_paths, val_label_paths, val_rois)
else:
    assert preconfigured_dataset in dataset_names, f"Invalid pre-configured dataset: {preconfigured_dataset}, choose one of {dataset_names}."
    if preconfigured_dataset in ("isbi2012", "vnc-mitos") and len(patch_shape) == 2:
        patch_shape = (1,) + patch_shape

assert len(patch_shape) in (2, 3)

# CONFIGURE ME

# Whether to add a foreground channel (1 for all labels that are not zero) to the target.
foreground = False
# Whether to add affinity channels (= directed boundaries) or a boundary channel to the target.
# Note that you can choose at most of these two options.
affinities = False
boundaries = False

# the pixel offsets that are used to compute the affinity channels
offsets = [[-1, 0], [0, -1], [-3, 0], [0, -3], [-9, 0], [0, -9]]

assert not (affinities and boundaries), "Predicting both affinities and boundaries is not supported"

label_transform, label_transform2 = None, None
if affinities:
    label_transform2 = torch_em.transform.label.AffinityTransform(
        offsets=offsets, add_binary_target=foreground, add_mask=True
    )
elif boundaries:
    label_transform = torch_em.transform.label.BoundaryTransform(
        add_binary_target=foreground
    )
elif foreground:
    label_transform = torch_em.transform.label.labels_to_binary

# Set batch size
batch_size=8


# Load datasets
kwargs = dict(
    ndim=2, patch_shape=patch_shape, batch_size=batch_size,
    label_transform=label_transform, label_transform2=label_transform2
)

ds = preconfigured_dataset

if ds is None:
    train_loader = torch_em.default_segmentation_loader(
        train_data_paths, data_key, train_label_paths, label_key,
        rois=train_rois, **kwargs
    )
    val_loader = torch_em.default_segmentation_loader(
        val_data_paths, data_key, val_label_paths, label_key,
        rois=val_rois, **kwargs
    )
else:
    kwargs.update(dict(download=False))
    if ds == "livecell":
        train_loader = torchem_data.get_livecell_loader(data_paths, split="train", **kwargs)
        val_loader = torchem_data.get_livecell_loader(data_paths, split="val", **kwargs)

assert train_loader is not None, "Something went wrong"
assert val_loader is not None, "Something went wrong"



# choose the number of samples to check per loader
n_samples = 4

print("Training samples")
check_loader(train_loader, n_samples, plt=True)
print("Validation samples")
check_loader(val_loader, n_samples, plt=True)