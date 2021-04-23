# Her2

The algorithm for extracting rois in this branch can extract high resolution rois (high magnification) from Warwick dataset
without requiring too much ram.

1. Download the script.
```
!wget https://raw.githubusercontent.com/Surayuth/Her2/crop/read_slide.py
```
2. Install these libraries.
```
!sudo apt-get install openslide-tools
!sudo apt-get install python-openslide
!pip install openslide-python
```
3. Read .ndpi image from `path`. We need to use `get_obj` to read .ndpi to OpenSlide object. Then, we use `get_img` to extract 
to extact PIL images from the specified `level`. 
```
obj = get_obj(path)
img = get_img(obj, level=5)
```


