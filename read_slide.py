import os
import cv2
import copy
import random
import numpy as np
import openslide as op
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# 0: download WSI
def download(src, dst, option=None):
    file = src.split('/')[-1]
    abspath = os.path.abspath(dst)
                  
    if not os.path.exists(os.path.join(dst, file)):
        os.system('cp' + ' ' + src + ' ' + dst)
    else:
        print(f'{file} is already exist in {abspath}...') 

    if option == 'x':
        print(f'extracting {file} to {abspath}...')
        os.system ('unzip' + ' -a ' + os.path.join(abspath, file))
        print(f'{file} is extracted in {abspath}...')
        
# 1: read image
def get_obj(path):
    return op.OpenSlide(path)

# 2.1: get image
def get_img(obj, lv):
    w = obj.level_dimensions[lv][0]
    h = obj.level_dimensions[lv][1]
    vector = [(0, 0), lv, (w, h)]
    return {'level':lv, 'image':np.array(obj.read_region(*vector))}

# 2.2: plot mask
def plot(mask):
    plt.figure(figsize=(10, 10))
    plt.imshow(mask['image'], cmap='gray')
    height, width = mask['image'].shape
    plt.title(f'level: {mask["level"]} | (width, height): {(width, height)}')

# 3.1: get mask
def get_mask(img, channels, ret=None):
    hsv_img = cv2.cvtColor(img['image'], cv2.COLOR_BGR2HSV)
    hs_img = np.zeros(hsv_img.shape[0:2])
    for c in channels:
        hs_img += hsv_img[:,:,c]
    hs_img = hs_img.astype(np.uint8)
    if ret == None:
        ret, thresh_img = cv2.threshold(hs_img, 0, 255, cv2.THRESH_OTSU)
    else:
        _, thresh_img = cv2.threshold(hs_img, ret, 255, cv2.THRESH_OTSU)
    return ret, {'level':img['level'], 'image':thresh_img}

# 3.2: (optional) crop mask
def crop_mask(mask, miny, minx):
    cropped = copy.deepcopy(mask)
    cropped['image'][0:miny, 0:minx] = 0
    return cropped 

# 3.3: (optional) get mask for different level
def convert_mask(mask, level_dims, to_level):
    new_w, new_h = level_dims[to_level]
    cvt_mask = {'level':to_level, 'image':mask['image']}
    cvt_mask['image'] = cv2.resize(mask['image'], (new_w, new_h), interpolation = cv2.INTER_NEAREST)
    return cvt_mask

# 3.4: (optional) remove control part
def remove_control(mask, rule='kmean'):
    image = np.zeros(mask['image'].shape)
    if rule == 'kmean':
        # calculate clusters
        Y, X = np.where(mask['image'] != [0])
        X = X.tolist()
        Y = Y.tolist()
        X_train = np.array([X, Y]).T

        # train km
        km = KMeans(n_clusters=3)
        km.fit(X_train)

        # get cluster center
        center = km.cluster_centers_

        # calculate distance
        dis = []
        for i in range(len(center)):
            tot = 0
            for j in range(len(center)):
                tot += np.linalg.norm(center[i]-center[j])
            dis.append(tot)

        # get the center of the control part
        control_idx = np.argmax(dis)

        # relate the coordinate of the center to its cluster number
        control_cluster = km.predict([center[control_idx]])[0]

        # delete indices of coordinates belonging to the control part
        indices = np.where(km.predict(X_train) != control_cluster)[0]
        new_X_train = X_train[indices,:]
        new_X, new_Y = new_X_train.T

        # transform the new coordinates to figure
        image[new_Y, new_X] = 255
   
    return {'level':mask['level'], 'image':image}

# 4.1: get rois of patches in mask
def get_rois(mask, size, thresh=0.3, show=False):
    height, width = mask['image'].shape
    numx = np.floor(width/size).astype(np.uint8)
    numy = np.floor(height/size).astype(np.uint8)

    coordinates = []
    if show:
        plt.figure(figsize=(10, 10))
        plt.imshow(mask['image'], cmap='gray')

    for y in range(numy):
        for x in range(numx):
            patch = mask['image'][size*y:size*(y+1), size*x:size*(x+1)]

            if np.sum(patch > 0) / size**2 > thresh:
                coordinates.append((size*y, size*x))
                if show:
                    plt.scatter(size*(x + 1/2), size*(y + 1/2), c='red')          
            else:
                if show:
                    plt.scatter(size*(x + 1/2), size*(y + 1/2), c='blue')
    if show:
        plt.title(f'level: {mask["level"]} | (width, height): {(width, height)}  | size: {size}')
        plt.show()    

    return {'level':mask['level'], 'size': size, 'coordinates':coordinates}
   
# 4.2: get coordinates of high res patches in each patch from 4.1
def convert_coordinates(coordinates, from_level, to_level):
    new_coordinates = []
    diff = from_level - to_level
    for coordinate in coordinates:
        y, x = coordinate
        new_y, new_x = y * 2**diff, x * 2**diff
        new_coordinates.append((new_y, new_x))
    return new_coordinates

def get_highres_rois(rois, to_level, magnify):
    '''
    ############################################################################
    # Extract high res rois from each low roi specified by each coordinate.
    ############################################################################
    # the number of new high res rois per each low res roi are varied, 
    # say #num * magnify = 2**diff
    # where #num = number of high res rois per each low res roi along x and y dirs.
    #   magnify = magnification (2^n where n = 0,1,2,..,diff)
    #   diff = from_level - to_level
    # For example, if diff = 4 and magnify = 2, then
    #   #num = 2**4 / 2 = 16
    # Thus, the number of high res rois in each low res roi = 16**2 
    ############################################################################
    '''
    from_level = rois['level']
    diff = from_level - to_level

    if magnify%2 != 0:
        raise Exception('magnification must be divisible by 2!')
    
    if magnify > 2**diff:
        raise Exception(f'magnification must be lower than {2**diff}!')

    # convert low res rois to high res rois
    coordinates = rois['coordinates']
    new_coordinates = convert_coordinates(coordinates, from_level, to_level)

    new_size = rois['size'] * magnify
    num = int(2**diff / magnify)
    smaller_coordinates = []
    for coordinate in new_coordinates:
        cornor_y, cornor_x = coordinate
        for y in range(1,num):
            for x in range(1,num):
                smaller_coordinates.append((cornor_y + new_size*y, 
                                            cornor_x + new_size*x))
 
    new_coordinates += smaller_coordinates

    return {'level':to_level, 'size': new_size, 'coordinates':new_coordinates}

# get absolute coordinates (coordinates with respect to level 1) 
def get_abs_coordinates(level_dims, level, coordinates): 
    w, h = level_dims[level]
    abs_w, abs_h = level_dims[0]
    abs_coordinates = []
    for coordinate in coordinates:
        y, x = coordinate
        abs_y, abs_x = y/h * abs_h, x/w * abs_w
        abs_coordinates.append((abs_y, abs_x))
    return np.rint(np.array(abs_coordinates)).astype(np.int32)

# 5: get patches from rois
def get_patches(obj, rois):
    level = rois['level']
    size = rois['size']
    coordinates = rois['coordinates']
    w, h = obj.level_dimensions[level]
    num_patches = len(rois['coordinates'])
    images = []
    abs_coordinates = get_abs_coordinates(obj.level_dimensions, level, coordinates)
    for abs_coordinate in abs_coordinates:
        y, x = abs_coordinate
        img = obj.read_region((x, y), level, (size, size))
        images.append(np.array(img))
    return {'level':level, 'size':size, 'images':images, 'abs':abs_coordinates}

# 6: remove outlier
def rand_imgs(images, ratio):
    # random patches to create a sample image
    num_img = len(images)  
    _, size, channels = images[0].shape
    sqrt_num = int(np.floor(np.sqrt(num_img * ratio)))
    rand_imgs = random.choices(images, k=int(sqrt_num**2))
    combined_img = np.zeros((sqrt_num*size, sqrt_num*size, channels))

    count = 0
    for y in range(sqrt_num):
        for x in range(sqrt_num):
            combined_img[size*y:size*(y+1), size*x:size*(x+1), :] = rand_imgs[count]
            count += 1
    combined_img = combined_img.astype(np.uint8)
    return rand_imgs, {'image':combined_img, 'level':'undefined'}


def remove_outliers(patches, ratio, channels, thresh=0.25):
    images = patches['images']
    size = patches['size']
    level = patches['level']
    # random patches to create a sample image
    _, combined_img = rand_imgs(images, ratio)

    # get threshold value
    ret,_ = get_mask(combined_img, channels)

    # remove outlier patches
    no_outliers = []
    for image in images:
        _, thresh_img = get_mask({'image':image, 'level':'undefined'}, channels)

        # check if we use disregrad this image
        if np.sum(thresh_img['image'] > 0) / size**2 > thresh:
            no_outliers.append(image)
    
    return {'level':level, 'size':size, 'images':no_outliers}

# 7: visualize 
def visual(patches, rows, cols, img_size):
    images = patches['images']
    if rows * cols < len(images):
        fig, axs = plt.subplots(rows, cols, figsize=(img_size, img_size))
        idx = 0
        for row in range(rows):
            for col in range(cols):
                axs[row][col].imshow(images[idx], cmap='gray')
                idx += 1
