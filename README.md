# Her2

The algorithm for extracting rois in this branch can extract high resolution rois (high magnification) from Warwick dataset
without requiring too much ram.

1. download the script
```
!wget https://raw.githubusercontent.com/Surayuth/Her2/main/read_slide.py
```
2. install these libraries 
```
!sudo apt-get install openslide-tools
!sudo apt-get install python-openslide
!pip install openslide-python
```
3. get tissues 
```
from read_slide import *
tissues = extract_tissues(path_to_image, from_level=5, to_level=3, size=224)
```
4. visualize tissues
```
visual(tissues, rows=2, cols=2, img_size=10)
```
![Screenshot from 2021-04-19 01-05-27](https://user-images.githubusercontent.com/66277085/115155880-0003cc00-a0ac-11eb-80a7-cebe897c67bc.png)

