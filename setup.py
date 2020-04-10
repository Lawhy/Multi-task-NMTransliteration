import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="multi-task-nmt-lawhy",
    version="0.1.1",
    author="Yuan (Lawrence) He",
    author_email="lawhy0729@gmail.com",
    description="Pytorch Implementation of the Multi-task Neural Machine Transliteration System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["torchtext>=0.5.0",
                      "torchvision>=0.5.0",
                      "python-Levenshtein>=0.12.0",
                      "pandas>=1.0.3"],
    url="https://github.com/Lawhy/Multi-task-NMT",
    packages=setuptools.find_packages(),
    package_data={
        "": ["*.tsv", "*.json", "*.xlsx"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
