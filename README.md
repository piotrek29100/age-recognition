
# Real time age recognition

## Dependencies

Face recognition package: https://github.com/ageitgey/face_recognition

Installing face_recognition on Windows:
1. Install CMake for Windows
2. Install Pillow, dlib, face_recognition
```python
pip install Pillow==4.0.0
pip install dlib
pip install face_recognition
```
## Image set preparation

1. Download image set form https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/ (https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar - imdb faces only)
2. Extract to main folder (.\\imdb_crop\\...)
3. Use convert_matlab_file.py code to convert imdb.mat (matlab format) to imdb.csv
4. Use prepare_images_sets.py code to prepare clean image set

## Model training

Use model_training.py code to train model. \
Basic configuration parameters:
* steps_per_epoch
* epochs
* frozen_layers
