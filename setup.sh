pip install gdown 
pip install tifffile
pip install albumentations
pip install segmentation_models_pytorch
gdown 'https://drive.google.com/uc?id=1yiqzoWtSGtR18TkHBcGhYkScS02R3vKH&export=download'
unzip -q EndoCV.zip -d EndoCV
rm -r EndoCV.zip
mkdir models logs