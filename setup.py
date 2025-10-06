from setuptools import setup, find_packages


setup(
    name="monai-multimodal",
    version="0.1.0",
    description="Multi-modal MONAI model with DICOM to NIfTI conversion (MRI/CT)",
    author="owner",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "monai>=1.3.0",
        "torch>=2.1.0",
        "numpy>=1.22",
        "nibabel>=5.0.0",
        "pydicom>=2.3.0",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "mm-convert=monai_multimodal.cli:main",
        ]
    },
)


