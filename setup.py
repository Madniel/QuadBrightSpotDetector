from setuptools import setup, find_packages

setup(
    name='brightness_patch_detector',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy==1.19.5',
        'opencv_python==4.8.0.76',
        'pytest==7.1.1',
        'setuptools==61.2.0'
    ],
)