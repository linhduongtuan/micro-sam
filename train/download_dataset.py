import numpy as np
import torch.nn as nn
import torch_em
import torch_em.data.datasets as torchem_data


dataset_names = [
    "axondeepseg",
    "bcss",
    "cem",
    "covid_if", 
    "cremi",
    "deepbacs",
    "dsb", 
    "hpa", 
    "isbi2012", 
    "kasthuri",
    "livecell",
    "lizard",
    "lucchi",
    "mitoem",
    "monusac",
    "monuseg",
    "mouse_embryo",
    "neurips_cell_seg",
    "nuc_mm",
    "pannuke",
    "plantseg",
    "platynereis",
    "pnas_arabidopsis",
    "snemi",
    "sponge_em",
    "tissuenet",
    "uro_cell"
    "vnc-mitos", 
    
]
preconfigured_dataset = "covid_if"

# Where to download the training data (the data will be downloaded only once).
# If you work in google colab you may want to adapt this path to be on your google drive, in order
# to not loose the data after each session.
download_folder = f"./data/{preconfigured_dataset}"

#
# use a custom dataset
#

# Create a custom dataset from local data by specifiying the paths for training data, training labels
# as well as validation data and validation labels
train_data_paths = []
val_data_paths = []
data_key = ""
train_label_paths = []
val_label_paths = []
label_key = ""
# example 1: Use training raw data and labels stored as tif-images in separate folders.
# The example is formulated using the data from the `dsb` dataset.

# train_data_paths = "./training_data/dsb/train/images"
# val_data_paths = "./training_data/dsb/test/images"
# data_key = "*.tif"
# train_label_paths = "./training_data/dsb/train/masks"
# val_label_paths = "./training_data/dsb/test/masks"
# label_key = "*.tif"
# patch_shape = (256, 256)

# example 2: Use training data and labels stored as a single stack in an hdf5 file.
# This example is formulated using the data from the `vnc-mitos` dataset,
# which stores the raw data in `/raw` and the labels in `/labels/mitochondria`.
# Note that we use roi's here to get separate training and val data from the same file.
# train_data_paths etc. can also be lists in order to train from multiple stacks.

# train_data_paths = train_label_paths = val_data_paths = val_label_paths = "./training_data/vnc-mitos/vnc_train.h5"
# data_key = "/raw"
# label_key = "/labels/mitochondria"
# train_rois = np.s_[:18, :, :]
# val_rois = np.s_[18:, :, :]
# patch_shape = (1, 512, 512)

#
# choose the patch shape
#

# This should be chosen s.t. it is smaller than the smallest image in your training data.
# If you are training from 3d data (data with a z-axis), you will need to specify the patch_shape
# as (1, shape_y, shape_x).

# In addition you can also specify region of interests for training using the normal python slice syntax
train_rois = None
val_rois = None
patch_shape = (512, 512)

dataset_names = [
    "covid_if", "dsb", "hpa", "isbi2012", "livecell", "vnc-mitos", "lucchi"
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

# CONFIGURE ME
batch_size = 1
loss = "dice"
metric = "dice"

def get_loss(loss_name):
    loss_names = ["bce", "ce", "dice"]
    if isinstance(loss_name, str):
        assert loss_name in loss_names, f"{loss_name}, {loss_names}"
        if loss_name == "dice":
            loss_function = torch_em.loss.DiceLoss()
        elif loss == "ce":
            loss_function = nn.CrossEntropyLoss()
        elif loss == "bce":
            loss_function = nn.BCEWithLogitsLoss()
    else:
        loss_function = loss_name
    
    # we need to add a loss wrapper for affinities
    if affinities:
        loss_function = torch_em.loss.LossWrapper(
            loss_function, transform=torch_em.loss.ApplyAndRemoveMask()
        )
    return loss_function


loss_function = get_loss(loss)
metric_function = get_loss(metric)

kwargs = dict(
    ndim=2, 
    patch_shape=patch_shape, 
    batch_size=batch_size,
    label_transform=label_transform, 
    label_transform2=label_transform2
)
ds = preconfigured_dataset

def download_dataset():
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
        if ds == 'axondeepseg':
            # choose name='sem' or name='tem'
            train_loader = torchem_data.get_axondeepseg_loader(download_folder, name='sem', **kwargs) 
            val_loader = torchem_data.get_axondeepseg_loader(download_folder, name='sem', **kwargs)
            print(f'Length of training set is {len(train_loader)}')
            print(f'Length of validation set set is {len(val_loader)}')
            return train_loader, val_loader
    
        elif ds == 'bcss':
            train_loader = torchem_data.get_bcss_loader(download_folder,  **kwargs)
            val_loader = torchem_data.get_bcss_loader(download_folder,  **kwargs)
            print(f'Length of training set is {len(train_loader)}')
            print(f'Length of validation set set is {len(val_loader)}')
            return train_loader, val_loader
        ## TODO: Don't work now (2023-11-24)
        elif ds == "cem": 
            train_loader = torchem_data.get_cem_loader(download_folder, split="train", **kwargs)
            val_loader = torchem_data.get_cem_loader(download_folder, split="val", **kwargs)
            print(f'Length of training set is {len(train_loader)}')
            print(f'Length of validation set set is {len(val_loader)}')
            return train_loader, val_loader
        
        elif ds == "covid_if":
            # use first 5 images for validation and the rest for training
            train_range, val_range = (5, None), (0, 5)
            train_loader = torchem_data.get_covid_if_loader(download_folder, sample_range=train_range, **kwargs)
            val_loader = torchem_data.get_covid_if_loader(download_folder, sample_range=val_range, **kwargs)
            print(f'Length of training set is {len(train_loader)}')
            print(f'Length of validation set set is {len(val_loader)}')
            return train_loader, val_loader
        
        elif ds == "cremi":
            train_roi, val_roi = np.s_[:18, :, :], np.s_[18:, :, :]
            train_loader = torchem_data.get_cremi_loader(download_folder, split="train", **kwargs)
            val_loader = torchem_data.get_cremi_loader(download_folder, split="train", **kwargs)
            print(f'Length of training set is {len(train_loader)}')
            print(f'Length of validation set set is {len(val_loader)}')
            return train_loader, val_loader
        
        elif ds == "deepbacs":
            train_loader = torchem_data.get_deepbacs_loader(download_folder, split="train", **kwargs)
            val_loader = torchem_data.get_deepbacs_loader(download_folder, split="test", **kwargs)
            print(f'Length of training set is {len(train_loader)}')
            print(f'Length of validation set set is {len(val_loader)}')
            return train_loader, val_loader
        
        elif ds == "dsb":
            train_loader = torchem_data.get_dsb_loader(download_folder, split="train", **kwargs)
            val_loader = torchem_data.get_dsb_loader(download_folder, split="train", **kwargs)
            print(f'Length of training set is {len(train_loader)}')
            print(f'Length of validation set set is {len(val_loader)}')
            return train_loader, val_loader
        
        elif ds == "hpa":
            train_loader = torchem_data.get_hpa_segmentation_loader(download_folder, split="train", **kwargs)
            val_loader = torchem_data.get_hpa_segmentation_loader(download_folder, split="val", **kwargs)
            print(f'Length of training set is {len(train_loader)}')
            print(f'Length of validation set set is {len(val_loader)}')
            return train_loader, val_loader
        
        elif ds == "isbi2012":
            assert not foreground, "Foreground prediction for the isbi neuron segmentation data does not make sense, please change these setings"
            train_roi, val_roi = np.s_[:28, :, :], np.s_[28:, :, :]
            train_loader = torchem_data.get_isbi_loader(download_folder, rois=train_roi, **kwargs)
            val_loader = torchem_data.get_isbi_loader(download_folder, rois=val_roi, **kwargs)
            print(f'Length of training set is {len(train_loader)}')
            print(f'Length of validation set set is {len(val_loader)}')
            return train_loader, val_loader
        
        elif ds == "kasthuri":
            #assert not foreground, "Foreground prediction for the isbi neuron segmentation data does not make sense, please change these setings"
            #train_roi, val_roi = np.s_[:28, :, :], np.s_[28:, :, :]
            train_loader = torchem_data.get_kasthuri_loader(download_folder, split='train', **kwargs)
            val_loader = torchem_data.get_kasthuri_loader(download_folder, split='test', **kwargs)
            print(f'Length of training set is {len(train_loader)}')
            print(f'Length of validation set set is {len(val_loader)}')
            return train_loader, val_loader
        
        elif ds == "livecell":
            train_loader = torchem_data.get_livecell_loader(download_folder, split="train", **kwargs)
            val_loader = torchem_data.get_livecell_loader(download_folder, split="val", **kwargs)
            print(f'Length of training set is {len(train_loader)}')
            print(f'Length of validation set set is {len(val_loader)}')
            return train_loader, val_loader
        
        elif ds == "lucchi":
            train_loader = torchem_data.get_lucchi_loader(download_folder, split="train", **kwargs)
            val_loader = torchem_data.get_lucchi_loader(download_folder, split="test", **kwargs)
            print(f'Length of training set is {len(train_loader)}')
            print(f'Length of validation set set is {len(val_loader)}')
            return train_loader, val_loader
        
        elif ds == "mitoem":
            train_loader = torchem_data.get_mitoem_loader(download_folder, splits="train", 
                                                          samples=("human", "rat"), **kwargs)
            val_loader = torchem_data.get_mitoem_loader(download_folder, splits="val", 
                                                        samples=("human", "rat"),
                                                         **kwargs)
            print(f'Length of training set is {len(train_loader)}')
            print(f'Length of validation set set is {len(val_loader)}')
            return train_loader, val_loader
        
        elif ds == "monusac":
            train_loader = torchem_data.get_monusac_loader(download_folder, split="train", **kwargs)
            val_loader = torchem_data.get_monusac_loader(download_folder, split="test", **kwargs)
            print(f'Length of training set is {len(train_loader)}')
            print(f'Length of validation set set is {len(val_loader)}')
            return train_loader, val_loader
        
        elif ds == "monuseg":
            train_loader = torchem_data.get_monuseg_loader(download_folder, split="train", **kwargs)
            val_loader = torchem_data.get_monuseg_loader(download_folder, split="test", **kwargs)
            print(f'Length of training set is {len(train_loader)}')
            print(f'Length of validation set set is {len(val_loader)}')
            return train_loader, val_loader
        
        elif ds == "mouse_embryo":
            train_loader = torchem_data.get_mouse_embryo_loader(download_folder,
                                                                name=("membrane", "nuclei"),
                                                                split="train", **kwargs)
            val_loader = torchem_data.get_mouse_embryo_loader(download_folder, 
                                                              name=("membrane", "nuclei"),
                                                              split="test", **kwargs)
            print(f'Length of training set is {len(train_loader)}')
            print(f'Length of validation set set is {len(val_loader)}')
            return train_loader, val_loader
        

        elif ds == "neurips_cell_seg":
            train_loader = torchem_data.get_neurips_cellseg_unsupervised_loader(download_folder,
                                                                **kwargs)
            val_loader = torchem_data.get_neurips_cellseg_unsupervised_loader(download_folder, 
                                                               **kwargs)
            print(f'Length of training set is {len(train_loader)}')
            print(f'Length of validation set set is {len(val_loader)}')
            return train_loader, val_loader

        elif ds == "nuc_mm":
            train_loader = torchem_data.get_nuc_mm_loader(download_folder, 
                                                          sample=("mouse", "zebrafish"),
                                                          split="train", **kwargs)
            val_loader = torchem_data.get_nuc_mm_loader(download_folder, 
                                                        sample=("mouse", "zebrafish"),
                                                        split="val", **kwargs)
            print(f'Length of training set is {len(train_loader)}')
            print(f'Length of validation set set is {len(val_loader)}')
            return train_loader, val_loader
        

        elif ds == "pannuke":
            train_roi, val_roi = np.s_[:18, :, :], np.s_[18:, :, :]
            train_loader = torchem_data.get_pannuke_loader(download_folder, 
                                                           folds=["fold_1", "fold_2", "fold_3"],
                                                           rois=train_roi,
                                                           **kwargs)
            val_loader = torchem_data.get_pannuke_loader(download_folder, 
                                                         folds=["fold_1", "fold_2", "fold_3"],
                                                         rois=train_roi,
                                                         **kwargs)
            print(f'Length of training set is {len(train_loader)}')
            print(f'Length of validation set set is {len(val_loader)}')
            return train_loader, val_loader
        
        elif ds == "plantseg":
            train_loader = torchem_data.get_plantseg_loader(download_folder, 
                                                           name='root',
                                                           split="train",
                                                           #name='nuclei',
                                                           #split="train",
                                                           #name='ovules',
                                                           #split="train",
                                                           **kwargs)
            val_loader = torchem_data.get_plantseg_loader(download_folder, 
                                                         name='root',
                                                         split="test",
                                                         #name='nuclei',
                                                         #split="train",
                                                         #name='ovules',
                                                        #split="test",
                                                         **kwargs)
            print(f'Length of training set is {len(train_loader)}')
            print(f'Length of validation set set is {len(val_loader)}')
            return train_loader, val_loader
        

        elif ds == "platynereis":
            train_roi, val_roi = np.s_[:18, :, :], np.s_[18:, :, :]
            train_loader = torchem_data.get_platynereis_nuclei_loader(download_folder, 
                                                          rois=train_rois,
                                                          split="train", **kwargs)
            val_loader = torchem_data.get_platynereis_nuclei_loader(download_folder, 
                                                        rois=val_rois,
                                                        split="val", **kwargs)
            print(f'Length of training set is {len(train_loader)}')
            print(f'Length of validation set set is {len(val_loader)}')
            return train_loader, val_loader

        elif ds == "snemi":
            train_loader = torchem_data.get_snemi_dataset(download_folder,
                                                                **kwargs)
            val_loader = torchem_data.get_snemi_dataset(download_folder, 
                                                               **kwargs)
            print(f'Length of training set is {len(train_loader)}')
            print(f'Length of validation set set is {len(val_loader)}')
            return train_loader, val_loader
        
        elif ds == "sponge_em":
            train_loader = torchem_data.get_snemi_dataset(download_folder,
                                                          mode=("semantic", "instances"),
                                                                **kwargs)
            val_loader = torchem_data.get_snemi_dataset(download_folder, 
                                                        mode=("semantic", "instances"),
                                                               **kwargs)
            print(f'Length of training set is {len(train_loader)}')
            print(f'Length of validation set set is {len(val_loader)}')
            return train_loader, val_loader
        
        elif ds == "tissuenet":
            train_loader = torchem_data.get_tissuenet_dataset(download_folder,
                                                          split="train",
                                                          raw_channel=("nucleus", "cell", "rgb"),
                                                          label_channel=("nucleus", "cell"),
                                                                **kwargs)
            val_loader = torchem_data.get_tissuenet_dataset(download_folder, 
                                                        split="test",
                                                        raw_channel=("nucleus", "cell", "rgb"),
                                                        label_channel=("nucleus", "cell"),
                                                               **kwargs)
            print(f'Length of training set is {len(train_loader)}')
            print(f'Length of validation set set is {len(val_loader)}')
            return train_loader, val_loader
        
        elif ds == "uro_cell":
            train_loader = torchem_data.get_uro_cell_dataset(download_folder,
                                                          mode=("semantic", "instances"),
                                                          target=("lyso", "golgi"),
                                                                **kwargs)
            val_loader = torchem_data.get_uro_cell_dataset(download_folder, 
                                                            target=("lyso", "golgi"),
                                                               **kwargs)
            print(f'Length of training set is {len(train_loader)}')
            print(f'Length of validation set set is {len(val_loader)}')
            return train_loader, val_loader
        

        # monuseg is not fully implemented yet
        # elif ds == "monuseg":
        #     train_loader = torchem_data.get
        elif ds == "vnc-mitos":
            train_roi, val_roi = np.s_[:18, :, :], np.s_[18:, :, :]
            train_loader = torchem_data.get_vnc_mito_loader(download_folder, rois=train_roi, **kwargs)
            val_loader = torchem_data.get_vnc_mito_loader(download_folder, rois=val_roi, **kwargs)
            print(f'Length of training set is {len(train_loader)}')
            print(f'Length of validation set set is {len(val_loader)}')
            return train_loader, val_loader
        
        

    assert train_loader is not None, "Something went wrong"
    assert val_loader is not None, "Something went wrong"