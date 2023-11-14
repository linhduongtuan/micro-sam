import numpy as np
import torch.nn as nn
import torch_em
import torch_em.data.datasets as torchem_data
from torch_em.model import UNet2d
from torch_em.util.debug import check_loader, check_trainer

# Use a pre-configured dataset
# Specify a pre-configured dataset. Set to `None` in order to specify the training data via file-paths instead.
preconfigured_dataset = None

# Where to download the training data (the data will be downloaded only once).
# If you work in google colab you may want to adapt this path to be on your google drive, in order
# to not loose the data after each session.
download_folder = f'./database/LM/{preconfigured_dataset}'

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

# To train the specialist model using with LM,
# They use 3 datasets, named as below!

dataset_names = [
    #"covid_if", 
    #"dsb", 
    #"hpa", 
    #"isbi2012", 
    "livecell", 
    #"vnc-mitos",
    "DeepBacs", 
    #"Lizard", 
    #"Mouse_Embryo", 
    #"NeuIPS22-CellSeg",
    #"Nucleus_dsb", 
    #"PlantSeg", 
    "TissueNet"
]


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
    kwargs.update(dict(download=True))
    if ds == "covid_if":
        # use first 5 images for validation and the rest for training
        train_range, val_range = (5, None), (0, 5)
        train_loader = torchem_data.get_covid_if_loader(download_folder, sample_range=train_range, **kwargs)
        val_loader = torchem_data.get_covid_if_loader(download_folder, sample_range=val_range, **kwargs)
    elif ds == "dsb": # Same as Nucleus_DSB
        train_loader = torchem_data.get_dsb_loader(download_folder, split="train", **kwargs)
        val_loader = torchem_data.get_dsb_loader(download_folder, split="train", **kwargs)
    elif ds == "hpa":
        train_loader = torchem_data.get_hpa_segmentation_loader(download_folder, split="train", **kwargs)
        val_loader = torchem_data.get_hpa_segmentation_loader(download_folder, split="val", **kwargs)
    elif ds == "isbi2012":
        assert not foreground, "Foreground prediction for the isbi neuron segmentation data does not make sense, please change these setings"
        train_roi, val_roi = np.s_[:28, :, :], np.s_[28:, :, :]
        train_loader = torchem_data.get_isbi_loader(download_folder, rois=train_roi, **kwargs)
        val_loader = torchem_data.get_isbi_loader(download_folder, rois=val_roi, **kwargs)
    elif ds == "livecell":
        train_loader = torchem_data.get_livecell_loader(download_folder, split="train", **kwargs)
        val_loader = torchem_data.get_livecell_loader(download_folder, split="val", **kwargs)
    # monuseg is not fully implemented yet
    # elif ds == "monuseg":
    #     train_loader = torchem_data.get
    elif ds == "vnc-mitos":
        train_roi, val_roi = np.s_[:18, :, :], np.s_[18:, :, :]
        train_loader = torchem_data.get_vnc_mito_loader(download_folder, rois=train_roi, **kwargs)
        val_loader = torchem_data.get_vnc_mito_loader(download_folder, rois=val_roi, **kwargs)

    elif ds == "DeepBacs":
        train_roi, val_roi = np.s_[:18, :, :], np.s_[18:, :, :] # Need to check
        train_loader = torchem_data.get_deepbacs_loader(download_folder, rois=train_roi, **kwargs)
        val_loader = torchem_data.get_deepbacs_loader(download_folder, rois=val_roi, **kwargs)


    elif ds == "Lizard":
        #train_roi, val_roi = np.s_[:18, :, :], np.s_[18:, :, :] # Need to check
        train_loader = torchem_data.get_lizard_loader(download_folder, rois=train_roi, **kwargs)
        val_loader = torchem_data.get_lizard_loader(download_folder, rois=val_roi, **kwargs)

    elif ds == "Mouse_Embryo":
        #train_roi, val_roi = np.s_[:18, :, :], np.s_[18:, :, :] # Need to check
        train_loader = torchem_data.get_mouse_embryo_loader(download_folder, rois=train_roi, **kwargs)
        val_loader = torchem_data.get_mouse_embryo_loader(download_folder, rois=val_roi, **kwargs) 

    elif ds == "NeuIPS22-CellSeg":
        #train_roi, val_roi = np.s_[:18, :, :], np.s_[18:, :, :] # Need to check
        train_loader = torchem_data.get_neurips_cellseg_unsupervised_loader(download_folder, rois=train_roi, **kwargs)
        val_loader = torchem_data.get_neurips_cellseg_unsupervised_loader(download_folder, rois=val_roi, **kwargs)

    elif ds == "Nucleus_dsb": # Same as dsb dataset
        #train_roi, val_roi = np.s_[:18, :, :], np.s_[18:, :, :] # Need to check
        train_loader = torchem_data.get_dsb_loader(download_folder, rois=train_roi, **kwargs)
        val_loader = torchem_data.get_dsb_loader(download_folder, rois=val_roi, **kwargs)

    elif ds == "PlantSeg":
        #train_roi, val_roi = np.s_[:18, :, :], np.s_[18:, :, :] # Need to check
        train_loader = torchem_data.get_plantseg_loader(download_folder, rois=train_roi, **kwargs)
        val_loader = torchem_data.get_plantseg_loader(download_folder, rois=val_roi, **kwargs)  

    elif ds == "TissueNet":
        #train_roi, val_roi = np.s_[:18, :, :], np.s_[18:, :, :] # Need to check
        train_loader = torchem_data.get_tissuenet_loader(download_folder, rois=train_roi, **kwargs)
        val_loader = torchem_data.get_tissuenet_loader(download_folder, rois=val_roi, **kwargs)        

assert train_loader is not None, "Something went wrong"
assert val_loader is not None, "Something went wrong"



# choose the number of samples to check per loader
n_samples = 4

print("Training samples")
check_loader(train_loader, n_samples, plt=True)
print("Validation samples")
check_loader(val_loader, n_samples, plt=True)