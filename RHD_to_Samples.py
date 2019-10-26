from __future__ import print_function, unicode_literals

import pickle
import os
import numpy as np
import cv2

"""https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/"""
def crop_rect(img, rect):
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))
    # get row and col num in img
    height, width = img.shape[0], img.shape[1]
    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height))
    # now rotated rectangle becomes vertical and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)
    return img_crop, img_rot
"""END ATTRIBUTION SECTION"""
def crop_square(img, rect):
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))
    size = (max(size), )*2
    # get row and col num in img
    height, width = img.shape[0], img.shape[1]
    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height))
    # now rotated rectangle becomes vertical and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)
    return img_crop, img_rot
def crop_square_and_coords(img, rect, coords):
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))
    size = (max(size), )*2
    # get row and col num in img
    height, width = img.shape[0], img.shape[1]
    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height))
    # now rotated rectangle becomes vertical and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    # rotate coords
    coords = coords[:,0]
    coords_rot = np.dot(coords, M[:,:2].T) + M[:,2].T
    crop_offset = np.array([center[0] - size[0]/2.0, center[1] - size[1]/2.0])
    coords_rot -= crop_offset

    return img_crop, img_rot, coords_rot

# chose between training and evaluation set
set = 'training'
# set = 'evaluation'


# auxiliary function
def depth_two_uint8_to_float(top_bits, bottom_bits):
    """ Converts a RGB-coded depth into float valued depth. """
    depth_map = (top_bits * 2**8 + bottom_bits).astype('float32')
    depth_map /= float(2**16 - 1)
    return depth_map

# load annotations of this set
with open(os.path.join(set, 'anno_%s.pickle' % set), 'rb') as fi:
    anno_all = pickle.load(fi)

# avg_width = []
# avg_height = []
# skipped = 0
# total = 0

# iterate samples of the set
for sample_id, anno in anno_all.items():
    # load data
    image = cv2.imread(os.path.join(set, 'color', '%.5d.png' % sample_id), cv2.IMREAD_ANYCOLOR)#cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(os.path.join(set, 'mask', '%.5d.png' % sample_id), cv2.IMREAD_ANYCOLOR)#cv2.IMREAD_UNCHANGED)
    depth = cv2.imread(os.path.join(set, 'depth', '%.5d.png' % sample_id), cv2.IMREAD_ANYCOLOR)

    # process rgb coded depth into float: top bits are stored in red, bottom in green channel
    depth = depth_two_uint8_to_float(depth[:, :, 2], depth[:, :, 1])  # depth in meters from the camera

    # get info from annotation dictionary
    kp_coord_uv = anno['uv_vis'][:, :2]  # u, v coordinates of 42 hand keypoints, pixel
    kp_visible = (anno['uv_vis'][:, 2] == 1)  # visibility of the keypoints, boolean
    kp_coord_xyz = anno['xyz']  # x, y, z coordinates of the keypoints, in meters
    camera_intrinsic_matrix = anno['K']  # matrix containing intrinsic parameters

    # Project world coordinates into the camera frame
    kp_coord_uv_proj = np.matmul(kp_coord_xyz, np.transpose(camera_intrinsic_matrix))
    kp_coord_uv_proj = kp_coord_uv_proj[:, :2] / kp_coord_uv_proj[:, 2:]

    # Skip samples that contain more than one hand, as they may overlap
    visible_hands = []
    for i in range(len(kp_visible)//2):
        if kp_visible[i]:
            visible_hands.append('left')
            break
    for i in range(len(kp_visible)//2, len(kp_visible)):
        if kp_visible[i]:
            visible_hands.append('right')
            break
    if len(visible_hands) > 1:
        continue

    if 'left' in visible_hands:
        kp_coord_uv_proj = kp_coord_uv_proj[:21]
    else:
        kp_coord_uv_proj = kp_coord_uv_proj[21:]
    kp_coord_uv_proj = kp_coord_uv_proj[:, np.newaxis]


    # Generate hand mask
    mask[mask > 1] = 255
    mask[mask < 255] = 0
    mask = (mask/255.0).astype('bool')

    # Invert depth map and make everything black except hands
    depth_inverted = 1 - depth
    depth_inverted[~mask] = 0#np.nan
    equalized = cv2.equalizeHist((255*depth_inverted).astype('uint8'))
    # print(np.nanmax(depth_inverted))
    # depth_inverted -= np.nanmin(depth_inverted)
    # depth_inverted *= 255.0/np.nanmax(depth_inverted)
    # depth_inverted[np.isnan(depth_inverted)] = 0

    # Expand hand sizes so that we don't chop edges when cropping later
    kernel = np.ones((5, 5), dtype = 'uint8')
    contour_mask = cv2.dilate(255*mask.astype('uint8'), kernel, iterations=2)

    contours, hierarchy = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contours = [kp_coord_uv_proj.astype('int32')]
    largest_area = 0
    largest_contour = None
    for contour in contours:
        if cv2.contourArea(contour) > largest_area:
            largest_contour = contour
            largest_area = cv2.contourArea(contour)

    # Create rotated rectangle around hand
    rect = cv2.minAreaRect(largest_contour)
    box = np.int0(cv2.boxPoints(rect))
    cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

    # Crop and resize
    cropped, _, coords_cropped = crop_square_and_coords(equalized, rect, kp_coord_uv_proj)
    coords_cropped *= np.array([88.0/cropped.shape[0], 88.0/cropped.shape[1]])
    resized = cv2.resize(cropped, (88, 88))

    # Print out keypoints
    wrist = coords_cropped[0]
    thumb = coords_cropped[1:5]
    index = coords_cropped[5:9]
    middl = coords_cropped[9:13]
    rring = coords_cropped[13:17]
    pinky = coords_cropped[17:]

    # print('Wrist:\n{}'.format(wrist))
    # print('Thumb:\n{}'.format(thumb))
    # print('Index:\n{}'.format(index))
    # print('Middle:\n{}'.format(middl))
    # print('Ring:\n{}'.format(rring))
    # print('Pinky:\n{}'.format(pinky))

    # UNCOMMENT to save samples
    # np.save(os.path.join(set, 'MNIST-style', '%.5d.npy' % sample_id), resized)
    np.save(os.path.join(set, 'MNIST-style-coords', '%.5d.npy' % sample_id), coords_cropped)

    # Filter samples that aren't nearly square
    # total += 1
    # if abs(1 - cropped.shape[0]/cropped.shape[1]) > 0.5:
    #     skipped += 1
    #     print(skipped/total)
    #     continue

    # Compute average height and width
    # avg_width.append(cropped.shape[1]) ~ 91.4
    # avg_height.append(cropped.shape[0]) ~ 96.7
    # print('{}'.format(sum(avg_width)/len(avg_width)) + '    ' + '{}'.format(sum(avg_height)/len(avg_height)))

    # Display everything
    # cv2.imshow('image; bounded', image)
    # cv2.imshow('cropped', resized)

    # ch = cv2.waitKey(0)
    # if ch == 27:
    #     break

# cv2.destroyAllWindows()
