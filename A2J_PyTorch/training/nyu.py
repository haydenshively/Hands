import os
import cv2
import json
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
import scipy.io as scio

from util import pixel2world, world2pixel
from augmentation import RandomErasing
from models.a2j_post_process import A2JPostProcess
from models.a2j_loss import A2JLoss


'''----------------------------------------------------------------------------------------------------------------------------------------------------------MANUAL PARAMETERS'''
MEAN = np.array(-4.55807423)
STD = np.array(20.69040591)
RANDOM_SEED = 12345
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
# Specify camera intrinsics
fx = 588.03
fy = -587.07
u0 = 320
v0 = 240
# Specify image directories
camera_id = 1
image_dir_train = '/hdd/datasets/hands/nyu_hand_dataset/train'
image_dir__test = '/hdd/datasets/hands/nyu_hand_dataset/test'
# Specify files containing coords of hand keypoints
keypoint_file_train = '/hdd/datasets/hands/nyu_hand_dataset/train/joint_data.mat'
keypoint_file__test = '/hdd/datasets/hands/nyu_hand_dataset/test/joint_data.mat'
# Specify output directory and files
save_dir = 'results'
result_file = 'NYU_batch_64_12345.txt'
'''----------------------------------------------------------------------------------------------------------------------------------------------------------JSON PARAMETERS'''
params = {}
with open('training/nyu_params.json') as f:
    params = json.load(f)
    print('Active Parameters:')
    print(json.dumps(params, indent=2))
    print('')
# Dataset
# we could pull this information from the dataset folders, but it takes a long
# time to count up all of the files. better just to store count in JSON
NUM_KEYPOINT = params['keypoints']
NUM_FRAME_TRAIN = params['frames']['training']
NUM_FRAME_TEST = params['frames']['testing']
# Preprocessing
IMG_WIDTH = params['preprocessing']['crop']['width']
IMG_HEIGHT = params['preprocessing']['crop']['height']
RAND_SHIFT_CROP = params['preprocessing']['rand_shift']['xy']
RAND_SHIFT_DEPTH = params['preprocessing']['rand_shift']['depth']# TODO not implemented
RAND_ROTATE = params['preprocessing']['rand_rotate']
RAND_SCALE = params['preprocessing']['rand_scale']
THRESH_XY = params['preprocessing']['thresh']['xy']
THRESH_DEPTH = params['preprocessing']['thresh']['depth']
# Training
BATCH_SIZE = params['training']['batch_size']
LEARNING_RATE = params['training']['learning_rate']
WEIGHT_DECAY = params['training']['weight_decay']
EPOCHS = params['training']['epochs']
# Model
REG_LOSS_FACTOR = params['model']['reg_loss_factor']
SPATIAL_FACTOR = params['model']['spatial_factor']
# Anchors
IMG_DIV = 28
ANCHOR_STRIDE = 16


'''----------------------------------------------------------------------------------------------------------------------------------------------------------LOADING COORDS'''
joints_to_use = np.zeros(NUM_KEYPOINT, dtype='bool')
if NUM_KEYPOINT == 36:
    joints_to_use[:] = True
elif NUM_KEYPOINT = 14:
    joints_to_use[[0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32]] = True
else:
    print('To use {} keypoints, please specify which of the 36 joints should be omitted')
# Load coords from train set
keypointsUVD_train = scio.loadmat(keypoint_file_train)['joint_uvd'].astype(np.float32)
keypointsUVD__test = scio.loadmat(keypoint_file__test)['joint_uvd'].astype(np.float32)
keypointsUVD_train = keypointsUVD_train[camera_id - 1]
keypointsUVD__test = keypointsUVD__test[camera_id - 1]
keypointsUVD_train = keypointsUVD_train[:, joints_to_use, :]
keypointsUVD__test = keypointsUVD__test[:, joints_to_use, :]

center_train = keypointsUVD_train.mean(axis=1)
center__test = keypointsUVD__test.mean(axis=1)
sizes_train = keypointsUVD_train[:,:,:2].ptp(axis=1).max(axis=1)/2.0 + THRESH_XY
sizes__test = keypointsUVD__test[:,:,:2].ptp(axis=1).max(axis=1)/2.0 + THRESH_XY
# Set bounding box corners
# --> top left
lefttop_pixel_train = np.zeros((center_train.shape[0], 2))
lefttop_pixel__test = np.zeros((center__test.shape[0], 2))
lefttop_pixel_train[:,0] = center_train[:,0] - sizes_train
lefttop_pixel__test[:,0] = center__test[:,0] - sizes__test
lefttop_pixel_train[:,1] = center_train[:,1] - sizes_train
lefttop_pixel__test[:,1] = center__test[:,1] - sizes__test
# --> bottom right
rightbottom_pixel_train = np.zeros((center_train.shape[0], 2))
rightbottom_pixel__test = np.zeros((center__test.shape[0], 2))
rightbottom_pixel_train[:,0] = center_train[:,0] + sizes_train
rightbottom_pixel__test[:,0] = center__test[:,0] + sizes__test
rightbottom_pixel_train[:,1] = center_train[:,1] + sizes_train
rightbottom_pixel__test[:,1] = center__test[:,1] + sizes__test


'''----------------------------------------------------------------------------------------------------------------------------------------------------------PREPROCESSING'''
def transform(img, label, matrix):
    # img: [H, W], label, [N,2]
    img_out = cv2.warpAffine(img,matrix,(IMG_WIDTH,IMG_HEIGHT))
    label_out = np.ones((NUM_KEYPOINT, 3))
    label_out[:,:2] = label[:,:2].copy()
    label_out = np.matmul(matrix, label_out.T)
    return img_out, label_out.T


def preprocess(index, img, keypointsUVD, center, lefttop_pixel, rightbottom_pixel, augment=True):
    # Generate augmentation parameters
    if augment:
        random_offset = np.random.randint(-RAND_SHIFT_CROP, +RAND_SHIFT_CROP, size=4)
        random_rotate = np.random.randint(-RAND_ROTATE, +RAND_ROTATE)
        random_scale = np.random.rand()*RAND_SCALE[0] + RAND_SCALE[1]
    else:
        random_offset = np.zeros(4, dtype='int')
        random_rotate = 0
        random_scale = 1
    matrix = cv2.getRotationMatrix2D((IMG_WIDTH/2, IMG_HEIGHT/2), random_rotate, random_scale)
    # Recompute bounding box, incorporating random_offset
    new_Xmin = max(lefttop_pixel[index,0] + random_offset[0], 0)
    new_Ymin = max(lefttop_pixel[index,1] + random_offset[1], 0)
    new_Xmax = min(rightbottom_pixel[index,0] + random_offset[2], img.shape[1] - 1)
    new_Ymax = min(rightbottom_pixel[index,1] + random_offset[3], img.shape[0] - 1)
    # DEAL WITH IMAGES
    # Crop to bounding box and stretch/squish to appropriate size
    cropped = img[int(new_Ymin):int(new_Ymax), int(new_Xmin):int(new_Xmax)].copy()
    resized = cv2.resize(cropped, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST).astype('float32')
    # This is like a bounding box in the Z direction.
    # Anything outside of the window gets set to the center pixel's color
    resized[np.where(resized >= center[index,2] + THRESH_DEPTH)] = center[index,2]
    resized[np.where(resized <= center[index,2] - THRESH_DEPTH)] = center[index,2]
    # Set center pixel's color to be 0, then stretch Z direction by random_scale
    resized = (resized - center[index,2]) * random_scale
    # Same thing, but for MEAN and STD of the dataset
    resized = (resized - MEAN) / STD# TODO how much does this impact results
    # DEAL WITH LABELS
    label_xy = np.ones((NUM_KEYPOINT, 2), dtype = 'float32')
    label_xy[:,0] = (keypointsUVD[index,:,0].copy() - new_Xmin) * IMG_WIDTH / (new_Xmax - new_Xmin)
    label_xy[:,1] = (keypointsUVD[index,:,1].copy() - new_Ymin) * IMG_HEIGHT / (new_Ymax - new_Ymin)
    # FINISH UP
    # Apply random rotation and scale to images and xy labels
    if augment: resized, label_xy = transform(resized, label_xy, matrix)
    # Create final arrays
    images_out = np.ones((IMG_HEIGHT, IMG_WIDTH, 1), dtype='float32')
    labels_out = np.ones((NUM_KEYPOINT, 3), dtype = 'float32')
    # Populate them
    images_out[:,:,0] = resized
    images_out = images_out.transpose(2, 0, 1)
    labels_out[:,1] = label_xy[:,0]
    labels_out[:,0] = label_xy[:,1]
    # ** (finally) Apply scale to z labels
    labels_out[:,2] = (keypointsUVD[index,:,2] - center[index,2]) * random_scale

    return torch.from_numpy(images_out), torch.from_numpy(labels_out)


'''----------------------------------------------------------------------------------------------------------------------------------------------------------DATA LOADER'''
class NYUDataset(torch.utils.data.Dataset):
    def __init__(self, dir, centers, lefttop_pixel, rightbottom_pixel, keypointsUVD, augment=True):
        self.dir = dir
        self.centers = centers
        self.lefttop_pixel = lefttop_pixel
        self.rightbottom_pixel = rightbottom_pixel
        self.keypointsUVD = keypointsUVD
        self.augment = augment

        self.randomErase = RandomErasing(probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0])

    def __getitem__(self, index):
        filename = 'depth_%d_%07d.png' % (camera_id, index + 1)

        depth = NYUDataset.imread_cv(os.path.join(self.dir, filename))
        data, label = preprocess(index, depth, self.keypointsUVD, self.centers, self.lefttop_pixel, self.rightbottom_pixel, self.augment)

        if self.augment: data = self.randomErase(data)
        return data, label

    def __len__(self):
        return len(self.centers)

    @staticmethod
    def imread_cv(path):
        x = cv2.imread(path, -1)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = np.left_shift(x[:,:,1].astype('uint32'), 8) + x[:,:,2].astype('uint32')
        return x.astype('float32')


'''----------------------------------------------------------------------------------------------------------------------------------------------------------TRAINING'''
dataset_train = NYUDataset(image_dir_train, center_train, lefttop_pixel_train, rightbottom_pixel_train, keypointsUVD_train, augment=True)
dataset__test = NYUDataset(image_dir__test, center__test, lefttop_pixel__test, rightbottom_pixel__test, keypointsUVD__test, augment=False)
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
dataloader__test = torch.utils.data.DataLoader(dataset__test, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)



def train(net, use_gpu=False):
    post_process = A2JPostProcess(None, None, [IMG_HEIGHT//IMG_DIV, IMG_WIDTH//IMG_DIV], ANCHOR_STRIDE, use_gpu=use_gpu)
    criterion = A2JLoss(None, None, [IMG_HEIGHT//IMG_DIV, IMG_WIDTH//IMG_DIV], ANCHOR_STRIDE, SPATIAL_FACTOR, use_gpu=use_gpu)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    logging.basicConfig(
        format='%(asctime)s %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        filename=os.path.join(save_dir, 'train.log'),
        level=logging.INFO
    )
    logging.info('======================================================')

    for epoch in range(EPOCHS):
        net = net.train()
        train_loss_add = 0.0
        Cls_loss_add = 0.0
        Reg_loss_add = 0.0

        for i, (img, label) in enumerate(dataloader_train):
            if use_gpu:
                img, label = img.cuda(), label.cuda()

            heads = net(img)
            optimizer.zero_grad()

            Cls_loss, Reg_loss = criterion(heads, label)

            loss = Cls_loss + Reg_loss*REG_LOSS_FACTOR
            loss.backward()
            optimizer.step()

            train_loss_add = train_loss_add + (loss.item())*len(img)
            Cls_loss_add = Cls_loss_add + (Cls_loss.item())*len(img)
            Reg_loss_add = Reg_loss_add + (Reg_loss.item())*len(img)

            if i%10 == 0:
                print('epoch: ',epoch, ' step: ', i, 'Cls_loss ',Cls_loss.item(), 'Reg_loss ',Reg_loss.item(), ' total loss ',loss.item())

        scheduler.step(epoch)

        train_loss_add = train_loss_add / NUM_FRAME_TRAIN
        Cls_loss_add = Cls_loss_add / NUM_FRAME_TRAIN
        Reg_loss_add = Reg_loss_add / NUM_FRAME_TRAIN
        print('Mean Loss per Sample:\t\t%f' % train_loss_add)
        print('\t>from anchor heatmap:\t%f' % Cls_loss_add)
        print('\t>from joint offsets:\t%f' % Reg_loss_add)

        Error_test = 0

        if (epoch % 1 == 0):
            net = net.eval()
            output = torch.FloatTensor()
            outputTrain = torch.FloatTensor()

            for i, (img, label) in tqdm(enumerate(dataloader__test)):
                with torch.no_grad():
                    if use_gpu:
                        img, label = img.cuda(), label.cuda()
                    heads = net(img)
                    pred_keypoints = post_process(heads)
                    output = torch.cat([output,pred_keypoints.data.cpu()], 0)

            result = output.cpu().data.numpy()
            Error_test = errorCompute(result, keypointsUVD__test, center__test)
            print('Test error for epoch ', epoch, ' is ', Error_test)
            saveNamePrefix = '%s/Model%d_' % (save_dir, epoch) + str(SPATIAL_FACTOR) + '_' + str(REG_LOSS_FACTOR) + '_%d_%dx%d_%d_%d' % (NUM_KEYPOINT, IMG_WIDTH, IMG_HEIGHT, IMG_DIV, ANCHOR_STRIDE)
            torch.save(net.state_dict(), saveNamePrefix + '.pth')

        # log
        logging.info('Epoch#%d: total loss=%.4f, Cls_loss=%.4f, Reg_loss=%.4f, Err_test=%.4f, lr = %.6f'
        %(epoch, train_loss_add, Cls_loss_add, Reg_loss_add, Error_test, scheduler.get_lr()[0]))



def test(net, use_gpu=False):
    net.load_state_dict(torch.load('results/fin_34_wetD_0.0001_depFact_0.5_RegFact_3_rndShft_5.pth', map_location=torch.device('cpu')))
    net.eval()

    post_process = A2JPostProcess(None, None, [IMG_HEIGHT//IMG_DIV, IMG_WIDTH//IMG_DIV], ANCHOR_STRIDE, use_gpu=use_gpu)

    output = torch.FloatTensor()
    for i, (img, label) in tqdm(enumerate(dataloader__test)):
        with torch.no_grad():
            if use_gpu:
                img, label = img.cuda(), label.cuda()
            heads = net(img)
            pred_keypoints = post_process(heads)
            output = torch.cat([output,pred_keypoints.data.cpu()], 0)


    result = output.cpu().data.numpy()
    writeTxt(result, center__test)
    error = errorCompute(result, keypointsUVD__test, center__test)
    print('Error:', error)


def errorCompute(source, target, center):
    assert np.shape(source)==np.shape(target), "source has different shape with target"

    Test1_ = source.copy()
    target_ = target.copy()
    Test1_[:,:,0] = source[:,:,1]
    Test1_[:,:,1] = source[:,:,0]
    Test1 = Test1_  # [x, y, z]

    for i in range(len(Test1_)):
        Xmin = max(lefttop_pixel__test[i,0], 0)
        Ymin = max(lefttop_pixel__test[i,1], 0)
        Xmax = min(rightbottom_pixel__test[i,0], u0*2 - 1)
        Ymax = min(rightbottom_pixel__test[i,1], v0*2 - 1)

        Test1[i,:,0] = Test1_[i,:,0]*(Xmax-Xmin)/IMG_WIDTH + Xmin  # x
        Test1[i,:,1] = Test1_[i,:,1]*(Ymax-Ymin)/IMG_HEIGHT + Ymin  # y
        Test1[i,:,2] = source[i,:,2] + center[i][2]

    labels = pixel2world(target_, fx, fy, u0, v0)
    outputs = pixel2world(Test1.copy(), fx, fy, u0, v0)

    errors = np.sqrt(np.sum((labels - outputs) ** 2, axis=2))

    return np.mean(errors)

def writeTxt(result, center):

    resultUVD_ = result.copy()
    resultUVD_[:, :, 0] = result[:,:,1]
    resultUVD_[:, :, 1] = result[:,:,0]
    resultUVD = resultUVD_  # [x, y, z]

    for i in range(len(result)):
        Xmin = max(lefttop_pixel__test[i,0], 0)
        Ymin = max(lefttop_pixel__test[i,1], 0)
        Xmax = min(rightbottom_pixel__test[i,0], u0*2 - 1)
        Ymax = min(rightbottom_pixel__test[i,1], v0*2 - 1)

        resultUVD[i,:,0] = resultUVD_[i,:,0]*(Xmax-Xmin)/IMG_WIDTH + Xmin
        resultUVD[i,:,1] = resultUVD_[i,:,1]*(Ymax-Ymin)/IMG_HEIGHT + Ymin
        resultUVD[i,:,2] = result[i,:,2] + center[i][2]

    resultReshape = resultUVD.reshape(len(result), -1)

    with open(os.path.join(save_dir, result_file), 'w') as f:
        for i in range(len(resultReshape)):
            for j in range(NUM_KEYPOINT*3):
                f.write(str(resultReshape[i, j])+' ')
            f.write('\n')

    f.close()
