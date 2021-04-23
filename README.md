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
to extact images from the specified `level`. Note that `img` is a dictionary with 2 keys: `image` (Image) and `level`.
```
from read_slide import *
obj = get_obj(path)
img = get_img(obj, level=5)
plt.imshow(img['image'])
```
4. Get a mask which we will use as a template for extracting Region of Interests (ROIs). The image (3 or 4 channels) will be converted to HSV color space
before being thresholding. `channels` denote the channels which we will use before thresholding. To use more than 1 channel, say HS, you can change the argument.
to `channels = [0, 1]`
```
ret, mask = get_mask(img, channels=[1])
plot(mask)
```
4.1 (Optional) Remove the control part from the mask. You can add you a new method for removing the control part yourself. The default option is `kmeans`.
```
new_mask = remove_control(mask)
plot(new_mask)
```
5. Get ROIs from the mask. This function will create a square grid over the mask where the length of each square is denoted by `size`. Note that this function will return a dictionary with 3 keys: `level`, `size` and `coordinates`. Note that`coordinates `(y, x) are the coordinates of the right cornors of squares.
```
rois = get_rois(new_mask, size=64, show=True)
```
6. Get high magnified ROIs from the coordinates for ROIs in step 5. In this step, we will draw a square grid inside each ROI extracted from step 5. The coordinates of the grids inside each ROI will be converted to be coordinates in high magnification images. The level of magnification increased is set by `to_level` , the lower, the better. `magnify` is the ratio between the length of each square grid after being magnified and the original length of each square grid. Note that the height and width of the images increase doubly for each `level` lowered. Thus, if `size=64`, `to_level=1` and `magnify = 8`, we get a relation `num * magnify` = `2 ** diff` where `num` is the number of grids along x and y directions in each ROI in step 5 and `diff = level - to_level`.
```
hm_rois = get_highmag_rois(rois, to_level=1, magnify=8)
```
7. Get patches (ndarray) from coordinates in ROIs. You can also extract patches `rois` too but the patches you get will have low magnification.
```
patches = get_patches(obj, hm_rois)
```
8. Separate the patches into 2 groups: the selected group have the tissue region more than  the threshold value `thresh`. `ratio` is related to the number of patches which will be sample to calculate an optimum binary threshold. See step 4 for the definition of `channels`.
```
filtered= filter_patches(patches, ratio=0.3, channels=[1], thresh=0.2)
```
