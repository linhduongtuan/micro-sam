[![DOC](https://shields.mitmproxy.org/badge/docs-pdoc.dev-brightgreen.svg)](https://computational-cell-analytics.github.io/micro-sam/)
[![Conda](https://anaconda.org/conda-forge/micro_sam/badges/version.svg)](https://anaconda.org/conda-forge/micro_sam)
[![codecov](https://codecov.io/gh/computational-cell-analytics/micro-sam/graph/badge.svg?token=7ETPP5CABP)](https://codecov.io/gh/computational-cell-analytics/micro-sam)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7919746.svg)](https://doi.org/10.5281/zenodo.7919746)

# SegmentAnything for Microscopy

Tools for segmentation and tracking in microscopy build on top of [SegmentAnything](https://segment-anything.com/).
Segment and track objects in microscopy images interactively with a few clicks!

We implement napari applications for:
- interactive 2d segmentation (Left: interactive cell segmentation)
- interactive 3d segmentation (Middle: interactive mitochondria segmentation in EM)
- interactive tracking of 2d image data (Right: interactive cell tracking)

<img src="https://github.com/computational-cell-analytics/micro-sam/assets/4263537/d04cb158-9f5b-4460-98cd-023c4f19cccd" width="256">
<img src="https://github.com/computational-cell-analytics/micro-sam/assets/4263537/dfca3d9b-dba5-440b-b0f9-72a0683ac410" width="256">
<img src="https://github.com/computational-cell-analytics/micro-sam/assets/4263537/aefbf99f-e73a-4125-bb49-2e6592367a64" width="256">

If you run into any problems or have questions regarding our tool please open an issue on Github or reach out via [image.sc](https://forum.image.sc/) using the tag `micro-sam` and tagging @constantinpape.


## Installation and Usage for the `micro_sam` repository

You can install `micro_sam` via conda:
```
conda install -c conda-forge micro_sam napari pyqt torch_em
```
Or 
```
conda env create -f <ENV>.yaml -n <ENV_NAME> # <ENV> is either environment_cpu.yaml or environment_gpu.yaml
conda activate <ENV_NAME>
pip install -e .
```
You can then start the `micro_sam` tools by running `$ micro_sam.annotator` in the command line.

For an introduction in how to use the napari based annotation tools check out [the video tutorials](https://www.youtube.com/watch?v=ket7bDUP9tI&list=PLwYZXQJ3f36GQPpKCrSbHjGiH39X4XjSO&pp=gAQBiAQB).
Please check out [the documentation](https://computational-cell-analytics.github.io/micro-sam/) for more details on the installation and usage of `micro_sam`.

## Installation and Usage for the `sam-hq` repository
### **Quick Installation via pip**
```
pip install segment-anything-hq
python
from segment_anything_hq import sam_model_registry
model_type = "<model_type>" #"vit_l/vit_b/vit_h/vit_tiny"
sam_checkpoint = "<path/to/checkpoint>"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
```

see specific usage example (such as vit-l) by running belowing command:
```
export PYTHONPATH=$(pwd)
python demo/demo_hqsam_pip_example.py
```


### **Standard Installation**
The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Clone the repository locally and install with

```
git clone https://github.com/SysCV/sam-hq.git
cd sam-hq; pip install -e .
```

The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format. `jupyter` is also required to run the example notebooks.

```
pip install opencv-python pycocotools matplotlib onnxruntime onnx timm
```

### Example conda environment setup
```bash
conda create --name sam_hq python=3.8 -y
conda activate sam_hq
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install opencv-python pycocotools matplotlib onnxruntime onnx timm

# under your working directory
git clone https://github.com/SysCV/sam-hq.git
cd sam-hq
pip install -e .
export PYTHONPATH=$(pwd)
```

### **Model Checkpoints**

Three HQ-SAM model versions of the model are available with different backbone sizes. These models can be instantiated by running

```
from segment_anything import sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
```

Download the provided trained model below and put them into the pretrained_checkpoint folder:
```
mkdir pretrained_checkpoint
``` 

Click the links below to download the checkpoint for the corresponding model type. We also provide **alternative model downloading links** [here](https://github.com/SysCV/sam-hq/issues/5) or at [hugging face](https://huggingface.co/lkeab/hq-sam/tree/main).
- `vit_b`: [ViT-B HQ-SAM model.](https://drive.google.com/file/d/11yExZLOve38kRZPfRx_MRxfIAKmfMY47/view?usp=sharing)
- `vit_l`: [ViT-L HQ-SAM model.](https://drive.google.com/file/d/1Uk17tDKX1YAKas5knI4y9ZJCo0lRVL0G/view?usp=sharing)
- `vit_h`: [ViT-H HQ-SAM model.](https://drive.google.com/file/d/1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8/view?usp=sharing)
- `vit_tiny` (**Light HQ-SAM** for real-time need): [ViT-Tiny HQ-SAM model.](https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth)

### **Getting Started**

First download a [model checkpoint](#model-checkpoints). Then the model can be used in just a few lines to get masks from a given prompt:

```
from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
predictor = SamPredictor(sam)
predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_prompts>)
```

Additionally, see the usage examples in our [demo](/demo/demo_hqsam.py) , [colab notebook](https://colab.research.google.com/drive/1QwAbn5hsdqKOD5niuBzuqQX4eLCbNKFL?usp=sharing) and [automatic mask generator notebook](https://colab.research.google.com/drive/1dhRq4eR6Fbl-yl1vbQvU9hqyyeOidQaU?usp=sharing).

To obtain HQ-SAM's visual result:
```
python demo/demo_hqsam.py
```

To obtain baseline SAM's visual result. Note that you need to download original SAM checkpoint from [baseline-SAM-L model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) and put it into the pretrained_checkpoint folder.
```
python demo/demo_sam.py
```

To obtain Light HQ-SAM's visual result:
```
python demo/demo_hqsam_light.py
```

### **HQ-SAM Tuning and HQ-Seg44k Data**
We provide detailed training, evaluation, visualization and data downloading instructions in [HQ-SAM training](train/README.md). You can also replace our training data to obtain your own SAM in specific application domain (like medical, OCR and remote sensing).

Please change the current folder path to:
```
cd train
```
and then refer to detailed [readme instruction](train/README.md).

## Contributing

We welcome new contributions!

If you are interested in contributing to micro-sam, please see the [contributing guide](doc/contributing.md) and [developer documentation](doc/development.md). The first step is to [discuss your idea in anew issue](https://github.com/computational-cell-analytics/micro-sam/issues/new) with the current developers.

## Citation

If you are using this repository in your research please cite
- Our [preprint](https://doi.org/10.1101/2023.08.21.554208)
- and the original [Segment Anything publication](https://arxiv.org/abs/2304.02643).
- If you use a `vit-tiny` models please also cite [Mobile SAM](https://arxiv.org/abs/2306.14289).


## Related Projects

There are a few other napari plugins build around Segment Anything:
- https://github.com/MIC-DKFZ/napari-sam (2d and 3d support)
- https://github.com/JoOkuma/napari-segment-anything (only 2d support)
- https://github.com/hiroalchem/napari-SAM4IS

Compared to these we support more applications (2d, 3d and tracking), and provide finetuning methods and finetuned models for microscopy data.
[WebKnossos](https://webknossos.org/) also offers integration of SegmentAnything for interactive segmentation.


## Release Overview

**New in version 0.3.0**

- Support for ellipse and polygon prompts
- Support for automatic segmentation in 3d
- Training refactoring and speed-up of fine-tuning

**New in version 0.2.1 and 0.2.2**

- Several bugfixes for the newly introduced functionality in 0.2.0.

**New in version 0.2.0**

- Functionality for training / finetuning and evaluation of Segment Anything Models
- Full support for our finetuned segment anything models
- Improvements of the automated instance segmentation functionality in the 2d annotator
- And several other small improvements

**New in version 0.1.1**

- Fine-tuned segment anything models for microscopy (experimental)
- Simplified instance segmentation menu
- Menu for clearing annotations

**New in version 0.1.0**

- We support tiling in all annotators to enable processing large images.
- Implement new automatic instance segmentation functionality:
    - That is faster.
    - Enables interactive update of parameters.
    - And also works for large images by making use of tiled embeddings.
- Implement the `image_series_annotator` for processing many images in a row.
- Use the data hash in pre-computed embeddings to warn if the input data changes.
- Create a simple GUI to select which annotator to start.
- And made many other small improvements and fixed bugs.

**New in version 0.0.2**

- We have added support for bounding box prompts, which provide better segmentation results than points in many cases.
- Interactive tracking now uses a better heuristic to propagate masks across time, leading to better automatic tracking results.
- And have fixed several small bugs.
