import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fadex",
    version="0.1.0",
    author="Lucas Greff Meneses",
    author_email="lucasgreffmeneses@usp.br",
    description="A feature attribution explainability method for dimensionality reduction algorithms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/greffao/fadex",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "plotly",
        "tqdm"
    ],
    extras_require={
        "gpu": [
            "cupy",
            "cuml"
        ]
    }
)
