import os
import numpy as np

import torch

import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch_em
import torch_em.data.datasets as datasets
from torch_em.data import MinInstanceSampler, ConcatDataset
from torch_em.transform.label import PerObjectDistanceTransform
from torch_em.transform.raw import normalize_percentile, normalize

from micro_sam.training import identity
from micro_sam.training.util import ResizeRawTrafo, ResizeLabelTrafo


def neurips_raw_trafo(raw):
    raw = datasets.neurips_cell_seg.to_rgb(raw)  # ensures 3 channels for the neurips data
    raw = normalize_percentile(raw)
    raw = np.mean(raw, axis=0)
    raw = normalize(raw)
    raw = raw * 255
    return raw


def to_8bit(raw):
    raw = normalize(raw)
    raw = raw * 255
    return raw

def get_label_transform(min_size=0):
        label_transform = PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False,
            foreground=True, instances=True, min_size=min_size
        )
        return label_transform

def get_concat_lm_datasets(input_path, patch_shape, split_choice):
    assert split_choice in ["train", "val"]

    label_dtype = torch.float32
    sampler = MinInstanceSampler()

    # def get_label_transform(min_size=0):
    #     label_transform = PerObjectDistanceTransform(
    #         distances=True, boundary_distances=True, directed_distances=False,
    #         foreground=True, instances=True, min_size=min_size
    #     )
    #     return label_transform

    def get_ctc_datasets(
        input_path, patch_shape, sampler, raw_transform, label_transform,
        ignore_datasets=["Fluo-N2DH-GOWT1", "Fluo-N2DL-HeLa"]
    ):
        all_ctc_datasets = []
        for dataset_name in datasets.ctc.CTC_CHECKSUMS["train"].keys():
            if dataset_name in ignore_datasets:
                continue

            all_ctc_datasets.append(
                datasets.get_ctc_segmentation_dataset(
                    path=os.path.join(input_path, "ctc"), dataset_name=dataset_name, patch_shape=(1, *patch_shape),
                    sampler=sampler, raw_transform=raw_transform, label_transform=label_transform, download=True
                )
            )
        return all_ctc_datasets

    _datasets = [
        # SegmentationDataset(
        #     image_dir=os.path.join(input_path, "he2dapi", f"images_{split_choice}"),
        #     mask_dir=os.path.join(input_path, "he2dapi", f"masks_{split_choice}"),
        #     patch_shape=patch_shape,
        #     sampler=sampler
        # ),
        SegmentationDataset(
            image_dir=os.path.join(input_path, "he2cd3_dapi_ki67", f"images_{split_choice}"),
            mask_dir=os.path.join(input_path, "he2cd3_dapi_ki67", f"masks_{split_choice}"),
            patch_shape=patch_shape,
            sampler=sampler
        ),
        # SegmentationDataset(
        #     image_dir=os.path.join(input_path, "he2ki67", f"images_{split_choice}"),
        #     mask_dir=os.path.join(input_path, "he2ki67", f"masks_{split_choice}"),
        #     patch_shape=patch_shape,
        #     sampler=sampler
        # ),
        # SegmentationDataset(
        #     image_dir=os.path.join(input_path, "he2cd3", f"images_{split_choice}"),
        #     mask_dir=os.path.join(input_path, "he2cd3", f"masks_{split_choice}"),
        #     patch_shape=patch_shape,
        #     sampler=sampler
        # ),
        # datasets.get_tissuenet_dataset(
        #     path=os.path.join(input_path, "tissuenet"), split=split_choice, download=True, patch_shape=patch_shape,
        #     raw_channel="rgb", label_channel="cell", sampler=sampler, label_dtype=label_dtype,
        #     raw_transform=ResizeRawTrafo(patch_shape), label_transform=ResizeLabelTrafo(patch_shape, min_size=0),
        #     n_samples=1000 if split_choice == "train" else 100
        # ),
        # datasets.get_livecell_dataset(
        #     path=os.path.join(input_path, "livecell"), split=split_choice, patch_shape=patch_shape,
        #     download=True, label_transform=get_label_transform(), sampler=sampler,
        #     label_dtype=label_dtype, raw_transform=identity
        # ),
        # datasets.get_deepbacs_dataset(
        #     path=os.path.join(input_path, "deepbacs"), split=split_choice, patch_shape=patch_shape,
        #     raw_transform=to_8bit, label_transform=get_label_transform(), label_dtype=label_dtype,
        #     download=False, sampler=MinInstanceSampler(min_num_instances=4)
        # ),
        # datasets.get_neurips_cellseg_supervised_dataset(
        #     root=os.path.join(input_path, "neurips-cell-seg"), split=split_choice,
        #     patch_shape=patch_shape, raw_transform=neurips_raw_trafo, label_transform=get_label_transform(),
        #     label_dtype=label_dtype, sampler=MinInstanceSampler(min_num_instances=3)
        # ),
        # datasets.get_dsb_dataset(
        #     path=os.path.join(input_path, "dsb"), split=split_choice if split_choice == "train" else "test",
        #     patch_shape=patch_shape, label_transform=get_label_transform(), sampler=sampler,
        #     label_dtype=label_dtype, download=True, raw_transform=identity
        # ),
        # datasets.get_plantseg_dataset(
        #     path=os.path.join(input_path, "plantseg"), name="root", sampler=MinInstanceSampler(min_num_instances=10),
        #     patch_shape=(1, *patch_shape), download=False, split=split_choice, ndim=2, label_dtype=label_dtype,
        #     raw_transform=ResizeRawTrafo(patch_shape, do_rescaling=False),
        #     label_transform=ResizeLabelTrafo(patch_shape, min_size=0),
        #     n_samples=1000 if split_choice == "train" else 100
        # ),
    ]
    # if split_choice == "train":
    #     _datasets += get_ctc_datasets(
    #         input_path, patch_shape, sampler, raw_transform=to_8bit, label_transform=get_label_transform()
    #     )

    generalist_dataset = ConcatDataset(*_datasets)

    # increasing the sampling attempts for the neurips cellseg dataset
    # generalist_dataset.datasets[3].max_sampling_attempts = 5000

    return generalist_dataset


def get_generalist_lm_loaders(input_path, patch_shape=(256, 256), batch_size_train=1, batch_size_val=1):
    """This returns the concatenated light microscopy datasets implemented in torch_em:
    https://github.com/constantinpape/torch-em/tree/main/torch_em/data/datasets
    It will automatically download all the datasets
        - expect NeurIPS CellSeg (Multi-Modal Microscopy Images) (https://neurips22-cellseg.grand-challenge.org/)

    NOTE: to remove / replace the datasets with another dataset, you need to add the datasets (for train and val splits)
    in `get_concat_lm_dataset`. The labels have to be in a label mask instance segmentation format.
    i.e. the tensors (inputs & masks) should be of same spatial shape, with each object in the mask having it's own ID.
    IMPORTANT: the ID 0 is reserved for background, and the IDs must be consecutive.
    """
    generalist_train_dataset = get_concat_lm_datasets(input_path, patch_shape, "train")
    generalist_val_dataset = get_concat_lm_datasets(input_path, patch_shape, "val")
    train_loader = torch_em.get_data_loader(generalist_train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=16)
    val_loader = torch_em.get_data_loader(generalist_val_dataset, batch_size=batch_size_val, shuffle=True, num_workers=16)
    return train_loader, val_loader




class ShuffleAttributeDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        shuffle = kwargs.pop('shuffle', False)
        super().__init__(*args, **kwargs)
        self.shuffle = shuffle


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, patch_shape, sampler=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_shape = patch_shape
        self.sampler = sampler
        
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        
        # Add ndim attribute
        self.ndim = 2  # Assuming 2D images
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Assuming masks are grayscale
        
        # Convert to numpy arrays
        image = np.array(image)
        mask = np.array(mask)
        
        # Apply transformations similar to other datasets
        image = neurips_raw_trafo(image)
        mask = get_label_transform()(mask)
        
        # Apply sampling if sampler is provided
        if self.sampler is not None:
            sample = self.sampler(image, mask)
            if isinstance(sample, bool):
                if not sample:
                    return self.__getitem__(np.random.randint(len(self)))
            else:
                image, mask = sample
        
        # Ensure correct shapes and types
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()
        
        # Ensure the shapes match patch_shape and have consistent number of channels
        if image.dim() == 2:
            image = image.unsqueeze(0)  # Add channel dimension if it's missing
        if image.shape[-2:] != self.patch_shape:
            image = F.interpolate(image.unsqueeze(0), size=self.patch_shape, mode='bilinear', align_corners=False).squeeze(0)
        
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # Add channel dimension if it's missing
        if mask.shape[-2:] != self.patch_shape:
            mask = F.interpolate(mask.unsqueeze(0), size=self.patch_shape, mode='nearest').squeeze(0)
        
        # Ensure both image and mask have the same number of channels (e.g., 1)
        if image.shape[0] != 1:
            image = image.mean(dim=0, keepdim=True)  # Convert to grayscale if it's not
        
        return image, mask

def get_data_loaders(data_dir, batch_size=32, num_workers=4, patch_shape=(256, 256)):
    train_dataset = SegmentationDataset(
        image_dir=os.path.join(data_dir, 'images_train'),
        mask_dir=os.path.join(data_dir, 'masks_train'),
        patch_shape=patch_shape,
        sampler=MinInstanceSampler()
    )
    
    test_dataset = SegmentationDataset(
        image_dir=os.path.join(data_dir, 'images_test'),
        mask_dir=os.path.join(data_dir, 'masks_test'),
        patch_shape=patch_shape,
        sampler=MinInstanceSampler()
    )
    
    train_loader = torch_em.get_data_loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch_em.get_data_loader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader



if __name__ == "__main__":
    base_dir = "/proj/aicell/users/x_liduo/seg/micro-sam/finetuning/generalists/training/light_microscopy/datasets"
    train_loader, test_loader = get_generalist_lm_loaders(base_dir, patch_shape=256)

    
    # Print some information about the data loaders
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    # Get a batch of training data
    images, masks = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Masks shape: {masks.shape}")
