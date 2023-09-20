from setuptools import setup, find_packages

setup(
    name='brightness_patch_detector',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'opencv_python',
        'pytest',
        'setuptools'
    ],
)