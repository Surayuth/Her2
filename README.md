# Her2
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
tissues = extract_tissues(path_to_image, from_level=5, to_level=3, size=224)
```
4. visualize tissues
```
visual(tissues, rows=2, cols=2, img_size=10)
```

