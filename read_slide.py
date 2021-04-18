import cv2
import copy
import numpy as np
import openslide as op
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# complete pipeline

class mask_config:
    thresh = 0.2
# 1: read image
def get_obj(path):
    return op.OpenSlide(path)

# 2.1: get image
def get_img(obj, lv, orig, wh):
    w, h = wh
    if w == -1:
        w = obj.level_dimensions[lv][0]
    
    if h == -1:
        h = obj.level_dimensions[lv][1]
    vector = [orig, lv, (w, h)]
    img = {'level':lv, 'image':np.array(obj.read_region(*vector))}
    return img

# 2.2: plot mask
def plot(mask):
    plt.figure(figsize=(10, 10))
    plt.imshow(mask['image'], cmap='gray')
    height, width = mask['image'].shape
    plt.title(f'level: {mask["level"]} | (width, height): {(width, height)}')

# 3.1: get mask
def get_mask(img):
    hsv_img = cv2.cvtColor(img['image'], cv2.COLOR_BGR2HSV)
    hs_img = hsv_img[:,:,0] + hsv_img[:,:,1]
    _, thresh_img = cv2.threshold(hs_img, 0, 255, cv2.THRESH_OTSU)
    mask = {'level':img['level'], 'image':thresh_img}
    return mask

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
    new_mask = {'level':mask['level'], 'image':np.zeros(mask['image'].shape)}
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
        new_mask['image'][new_Y, new_X] = 255
   
    return new_mask

# 4.1: get patches
def get_patches(mask, size, show=False):
    height, width = mask['image'].shape
    thresh = mask_config.thresh
    numx = np.floor(width/size).astype(np.uint8)
    numy = np.floor(height/size).astype(np.uint8)

    patches = {'level':mask['level'], 'size': size, 'coordinates':[], 'images':[]}
    if show:
        plt.figure(figsize=(10, 10))
        plt.imshow(mask['image'], cmap='gray')

    for y in range(numy):
        for x in range(numx):
            patch = mask['image'][size*y:size*(y+1), size*x:size*(x+1)]

            if np.sum(patch > 0) / size**2 > thresh:
                patches['coordinates'].append((size*y, size*x))
                patches['images'].append(patch)
                if show:
                    plt.scatter(size*(x + 1/2), size*(y + 1/2), c='red')          
            else:
                if show:
                    plt.scatter(size*(x + 1/2), size*(y + 1/2), c='blue')
    if show:
        plt.title(f'level: {mask["level"]} | (width, height): {(width, height)}  | size: {size}')
        plt.show()    

    return patches

# 5: get tissue from masked patches
def get_tissues(img, patches):
    if img['level'] != patches['level']:
        raise Exception('Levels of the image and mask are inconsistent.')

    tissues = {'level':img['level'], 'images':[]}
    size = patches['size']
    num_patches = len(patches['coordinates'])
    for idx in range(num_patches):
        y, x = patches['coordinates'][idx]
        tissues['images'].append(img['image'][y:y+size, x:x+size, :])
    return tissues

# 5: visualize 
def visual(patches, rows, cols, img_size):
    images = patches['images']
    if rows * cols < len(images):
        fig, axs = plt.subplots(rows, cols, figsize=(img_size, img_size))
        idx = 0
        for row in range(rows):
            for col in range(cols):
                axs[row][col].imshow(images[idx], cmap='gray')
                idx += 1

# complete pipeline
def extract_patches(path, from_level, to_level, size, show=False):
    obj = get_obj(path)
    level_dims = obj.leve_dimensions
    img = get_img(obj, from_level, (0, 0), (-1, -1))
    
    # normal mask (A)
    mask = get_mask(img)

    # remove control part
    new_mask = remove_control(mask)

    # mask with different size (new size of crop A)
    cvt_mask = convert_mask(new_mask, level_dims, to_level)

    # get patches of the new cropped mask 
    cvt_patches = get_patches(cvt_mask, size, show)

    return cvt_patches
