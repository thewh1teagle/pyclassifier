from setuptools import setup, find_packages

setup(
    name="hashtron",
    version="0.0.9",
    packages=find_packages(),
    install_requires=[
        "requests==2.32.3",
    ],
    author="Neurlang Project",
    author_email="77860779+neurlang@users.noreply.github.com",
    description="A Python inference-only engine for the hashtron binary classifier",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/neurlang/pyclassifier",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
