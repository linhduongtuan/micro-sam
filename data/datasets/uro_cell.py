import os
import warnings
from glob import glob
from shutil import rmtree

import h5py
import torch_em
from . import util


URL = "https://github.com/MancaZerovnikMekuc/UroCell/archive/refs/heads/master.zip"
CHECKSUM = "a48cf31b06114d7def642742b4fcbe76103483c069122abe10f377d71a1acabc"


def _require_urocell_data(path, download):
    if os.path.exists(path):
        return path

    # add nifti file format support in elf by wrapping nibabel?
    import nibabel as nib

    # download and unzip the data
    os.makedirs(path)
    tmp_path = os.path.join(path, "uro_cell.zip")
    util.download_source(tmp_path, URL, download, checksum=CHECKSUM)
    util.unzip(tmp_path, path, remove=True)

    root = os.path.join(path, "UroCell-master")

    files = glob(os.path.join(root, "data", "*.nii.gz"))
    files.sort()
    for data_path in files:
        fname = os.path.basename(data_path)
        data = nib.load(data_path).get_fdata()

        out_path = os.path.join(path, fname.replace("nii.gz", "h5"))
        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=data, compression="gzip")

            # check if we have any of the organelle labels for this volume
            # and also copy them if yes
            fv_path = os.path.join(root, "fv", "instance", fname)
            if os.path.exists(fv_path):
                fv = nib.load(fv_path).get_fdata().astype("uint32")
                assert fv.shape == data.shape
                f.create_dataset("labels/fv", data=fv, compression="gzip")

            golgi_path = os.path.join(root, "golgi", "precise", fname)
            if os.path.exists(golgi_path):
                golgi = nib.load(golgi_path).get_fdata().astype("uint32")
                assert golgi.shape == data.shape
                f.create_dataset("labels/golgi", data=golgi, compression="gzip")

            lyso_path = os.path.join(root, "lyso", "instance", fname)
            if os.path.exists(lyso_path):
                lyso = nib.load(lyso_path).get_fdata().astype("uint32")
                assert lyso.shape == data.shape
                f.create_dataset("labels/lyso", data=lyso, compression="gzip")

            mito_path = os.path.join(root, "mito", "instance", fname)
            if os.path.exists(mito_path):
                mito = nib.load(mito_path).get_fdata().astype("uint32")
                assert mito.shape == data.shape
                f.create_dataset("labels/mito", data=mito, compression="gzip")

    # clean up
    rmtree(root)


def _get_paths(path, target):
    label_key = f"labels/{target}"
    all_paths = glob(os.path.join(path, "*.h5"))
    all_paths.sort()
    paths = [path for path in all_paths if label_key in h5py.File(path, "r")]
    return paths, label_key


def get_uro_cell_dataset(
    path,
    target,
    patch_shape,
    download=False,
    offsets=None,
    boundaries=False,
    binary=False,
    **kwargs,
):
    """Dataset for the segmentation of mitochondria and other organelles in EM.

    This dataset is from the publication https://doi.org/10.1016/j.compbiomed.2020.103693.
    Please cite it if you use this dataset for a publication.
    """
    assert target in ("fv", "golgi", "lyso", "mito")
    _require_urocell_data(path, download)
    paths, label_key = _get_paths(path, target)

    assert (
        sum((offsets is not None, boundaries, binary)) <= 1
    ), f"{offsets}, {boundaries}, {binary}"
    if offsets is not None:
        if target in ("lyso", "golgi"):
            warnings.warn(
                f"{target} does not have instance labels, affinities will be computed based on binary segmentation."
            )
        # we add a binary target channel for foreground background segmentation
        label_transform = torch_em.transform.label.AffinityTransform(
            offsets=offsets, ignore_label=None, add_binary_target=True, add_mask=True
        )
        msg = "Offsets are passed, but 'label_transform2' is in the kwargs. It will be over-ridden."
        kwargs = util.update_kwargs(
            kwargs, "label_transform2", label_transform, msg=msg
        )
    elif boundaries:
        if target in ("lyso", "golgi"):
            warnings.warn(
                f"{target} does not have instance labels, boundaries will be computed based on binary segmentation."
            )
        label_transform = torch_em.transform.label.BoundaryTransform(
            add_binary_target=True
        )
        msg = "Boundaries is set to true, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = util.update_kwargs(kwargs, "label_transform", label_transform, msg=msg)
    elif binary:
        label_transform = torch_em.transform.label.labels_to_binary
        msg = "Binary is set to true, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = util.update_kwargs(kwargs, "label_transform", label_transform, msg=msg)

    raw_key = "raw"
    return torch_em.default_segmentation_dataset(
        paths, raw_key, paths, label_key, patch_shape, is_seg_dataset=True, **kwargs
    )


def get_uro_cell_loader(
    path,
    target,
    patch_shape,
    batch_size,
    download=False,
    offsets=None,
    boundaries=False,
    binary=False,
    **kwargs,
):
    """Dataloader for the segmentation of mitochondria and other organelles in EM. See 'get_uro_cell_dataset'."""
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    ds = get_uro_cell_dataset(
        path,
        target,
        patch_shape,
        download=download,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
