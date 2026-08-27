"""Setup script for the OpenMythos Recurrent-Depth Transformer package.

Install in editable (development) mode:

    pip install -e .
"""

import os

from setuptools import find_packages, setup

_HERE = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(_HERE, "README.md"), encoding="utf-8") as f:
    _LONG_DESCRIPTION = f.read()

setup(
    name="openmythos",
    version="1.0.0",
    description=(
        "OpenMythos: an enterprise-grade Recurrent-Depth Transformer (RDT) "
        "implementation with native CUDA / Triton / FlashAttention-3 support, "
        "BF16 / FP8 / NVFP4 precision, DDP/FSDP training and streaming FineWeb."
    ),
    long_description=_LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="OpenMythos Contributors",
    license="Apache-2.0",
    python_requires=">=3.11",
    packages=find_packages(include=["openmythos", "openmythos.*"]),
    install_requires=[
        # Heavy ML dependencies (torch, flash-attn, datasets, tiktoken, ...)
        # are intentionally NOT hard-pinned here to avoid conflicting with the
        # caller's CUDA-matched wheels. Install them via:
        #     pip install -r requirements.txt
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
        # Native HF shard-acquisition stack (default data path).
        "huggingface_hub>=0.23.0",
        "pyarrow>=14.0.0",
    ],
    extras_require={
        "data": ["datasets>=2.19.0", "tokenizers>=0.19.0", "tiktoken>=0.7.0"],
        "track": ["wandb>=0.17.0"],
        "flash": ["flash-attn>=2.8.0"],
        "fp8": ["torchao>=0.5.0"],
        "te": ["transformer_engine[pytorch]>=1.7.0"],
        "all": [
            "datasets>=2.19.0",
            "tokenizers>=0.19.0",
            "tiktoken>=0.7.0",
            "wandb>=0.17.0",
            "flash-attn>=2.8.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
