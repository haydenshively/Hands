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


'''-----------------------------------------------------------------------------ASSORTED CONSTANTS'''
MEAN = np.array(-0.66877532422628)
STD = np.array(28.329582076876047)
RANDOM_SEED = 12345
# Specify camera intrinsics
fx = 588.03
fy = -587.07
u0 = 320
v0 = 240
'''-----------------------------------------------------------------------------DATA LOCATION'''
# Specify image directories
image_dir_train = '/Volumes/T7 Touch/datasets/hands/NYU/train'
image_dir_test = '/Volumes/T7 Touch/datasets/hands/NYU/test'
# Specify files containing coords of hand centers
centers_train = '/Volumes/T7 Touch/datasets/hands/_centers/NYU/center_train_refined.txt'
centers_test = '/Volumes/T7 Touch/datasets/hands/_centers/NYU/center_test_refined.txt'
# Specify files containing coords of hand keypoints
keypoint_file_train = '/Volumes/T7 Touch/datasets/hands/NYU/train/joint_data.mat'
keypoint_file_test = '/Volumes/T7 Touch/datasets/hands/NYU/train/joint_data.mat'
'''-----------------------------------------------------------------------------JSON PARAMETERS'''
params = {}
with open('nyu_params.json') as f:
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
RAND_SHIFT_DEPTH = params['preprocessing']['rand_shift']['depth']
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





random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

save_dir = './result/NYU_batch_64_12345'

try:
    os.makedirs(save_dir)
except OSError:
    pass

model_dir = '../model/NYU.pth'
result_file = 'result_NYU.txt'





## loading GT keypoints and center points
keypointsUVD_test = scio.loadmat(keypoint_file_test)['keypoints3D'].astype(np.float32)
center_test = scio.loadmat(centers_test)['centre_pixel'].astype(np.float32)

centre_test_world = pixel2world(center_test.copy(), fx, fy, u0, v0)

centerlefttop_test = centre_test_world.copy()
centerlefttop_test[:,0,0] = centerlefttop_test[:,0,0]-THRESH_XY
centerlefttop_test[:,0,1] = centerlefttop_test[:,0,1]+THRESH_XY

centerrightbottom_test = centre_test_world.copy()
centerrightbottom_test[:,0,0] = centerrightbottom_test[:,0,0]+THRESH_XY
centerrightbottom_test[:,0,1] = centerrightbottom_test[:,0,1]-THRESH_XY

test_lefttop_pixel = world2pixel(centerlefttop_test, fx, fy, u0, v0)
test_rightbottom_pixel = world2pixel(centerrightbottom_test, fx, fy, u0, v0)


keypointsUVD_train = scio.loadmat(keypoint_file_train)['keypoints3D'].astype(np.float32)
center_train = scio.loadmat(centers_train)['centre_pixel'].astype(np.float32)
centre_train_world = pixel2world(center_train.copy(), fx, fy, u0, v0)

centerlefttop_train = centre_train_world.copy()
centerlefttop_train[:,0,0] = centerlefttop_train[:,0,0]-THRESH_XY
centerlefttop_train[:,0,1] = centerlefttop_train[:,0,1]+THRESH_XY

centerrightbottom_train = centre_train_world.copy()
centerrightbottom_train[:,0,0] = centerrightbottom_train[:,0,0]+THRESH_XY
centerrightbottom_train[:,0,1] = centerrightbottom_train[:,0,1]-THRESH_XY

train_lefttop_pixel = world2pixel(centerlefttop_train, fx, fy, u0, v0)
train_rightbottom_pixel = world2pixel(centerrightbottom_train, fx, fy, u0, v0)



def dataPreprocess(index, img, keypointsUVD, center, mean, std, lefttop_pixel, rightbottom_pixel, augment=True):

    imageOutputs = np.ones((IMG_HEIGHT, IMG_WIDTH, 1), dtype='float32')
    labelOutputs = np.ones((NUM_KEYPOINT, 3), dtype = 'float32')

    if augment:
        RandomOffset_1 = np.random.randint(-1*RAND_SHIFT_CROP,RAND_SHIFT_CROP)
        RandomOffset_2 = np.random.randint(-1*RAND_SHIFT_CROP,RAND_SHIFT_CROP)
        RandomOffset_3 = np.random.randint(-1*RAND_SHIFT_CROP,RAND_SHIFT_CROP)
        RandomOffset_4 = np.random.randint(-1*RAND_SHIFT_CROP,RAND_SHIFT_CROP)
        RandomOffsetDepth = np.random.normal(0, RAND_SHIFT_DEPTH, IMG_HEIGHT*IMG_WIDTH).reshape(IMG_HEIGHT,IMG_WIDTH)
        RandomOffsetDepth[np.where(RandomOffsetDepth < RAND_SHIFT_DEPTH)] = 0
        RandomRotate = np.random.randint(-1*RAND_ROTATE,RAND_ROTATE)
        RandomScale = np.random.rand()*RAND_SCALE[0]+RAND_SCALE[1]
        matrix = cv2.getRotationMatrix2D((IMG_WIDTH/2,IMG_HEIGHT/2),RandomRotate,RandomScale)
    else:
        RandomOffset_1, RandomOffset_2, RandomOffset_3, RandomOffset_4 = 0, 0, 0, 0
        RandomRotate = 0
        RandomScale = 1
        RandomOffsetDepth = 0
        matrix = cv2.getRotationMatrix2D((IMG_WIDTH/2,IMG_HEIGHT/2),RandomRotate,RandomScale)

    new_Xmin = max(lefttop_pixel[index,0,0] + RandomOffset_1, 0)
    new_Ymin = max(lefttop_pixel[index,0,1] + RandomOffset_2, 0)
    new_Xmax = min(rightbottom_pixel[index,0,0] + RandomOffset_3, img.shape[1] - 1)
    new_Ymax = min(rightbottom_pixel[index,0,1] + RandomOffset_4, img.shape[0] - 1)

    imCrop = img[int(new_Ymin):int(new_Ymax), int(new_Xmin):int(new_Xmax)].copy()

    imgResize = cv2.resize(imCrop, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)

    imgResize = np.asarray(imgResize,dtype = 'float32')  # H*W*C

    imgResize[np.where(imgResize >= center[index][0][2] + THRESH_DEPTH)] = center[index][0][2]
    imgResize[np.where(imgResize <= center[index][0][2] - THRESH_DEPTH)] = center[index][0][2]
    imgResize = (imgResize - center[index][0][2])*RandomScale

    imgResize = (imgResize - mean) / std

    ## label
    label_xy = np.ones((NUM_KEYPOINT, 2), dtype = 'float32')
    label_xy[:,0] = (keypointsUVD[index,:,0].copy() - new_Xmin)*IMG_WIDTH/(new_Xmax - new_Xmin) # x
    label_xy[:,1] = (keypointsUVD[index,:,1].copy() - new_Ymin)*IMG_HEIGHT/(new_Ymax - new_Ymin) # y

    if augment:
        def transform(img, label, matrix):
            # img: [H, W], label, [N,2]
            img_out = cv2.warpAffine(img,matrix,(IMG_WIDTH,IMG_HEIGHT))
            label_out = np.ones((NUM_KEYPOINT, 3))
            label_out[:,:2] = label[:,:2].copy()
            label_out = np.matmul(matrix, label_out.T)
            return img_out, label_out.T

        imgResize, label_xy = transform(imgResize, label_xy, matrix)

    imageOutputs[:,:,0] = imgResize

    labelOutputs[:,1] = label_xy[:,0]
    labelOutputs[:,0] = label_xy[:,1]
    labelOutputs[:,2] = (keypointsUVD[index,:,2] - center[index][0][2])*RandomScale   # Z

    imageOutputs = np.asarray(imageOutputs)
    imageNCHWOut = imageOutputs.transpose(2, 0, 1)  # [H, W, C] --->>>  [C, H, W]
    imageNCHWOut = np.asarray(imageNCHWOut)
    labelOutputs = np.asarray(labelOutputs)

    data, label = torch.from_numpy(imageNCHWOut), torch.from_numpy(labelOutputs)

    return data, label


######################   Pytorch dataloader   #################
class my_dataloader(torch.utils.data.Dataset):

    def __init__(self, ImgDir, center, lefttop_pixel, rightbottom_pixel, keypointsUVD, augment=True):

        self.ImgDir = ImgDir
        self.mean = MEAN
        self.std = STD
        self.center = center
        self.lefttop_pixel = lefttop_pixel
        self.rightbottom_pixel = rightbottom_pixel
        self.keypointsUVD = keypointsUVD
        self.augment = augment
        self.randomErase = random_erasing.RandomErasing(probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0])

    def __getitem__(self, index):

        depth = scio.loadmat(self.ImgDir + str(index+1) + '.mat')['depth']

        data, label = dataPreprocess(index, depth, self.keypointsUVD, self.center, self.mean, self.std, \
            self.lefttop_pixel, self.rightbottom_pixel, THRESH_XY, THRESH_DEPTH, self.augment)

        if self.augment:
            data = self.randomErase(data)

        return data, label

    def __len__(self):
        return len(self.center)


train_image_datasets = my_dataloader(image_dir_train, center_train, train_lefttop_pixel, train_rightbottom_pixel, keypointsUVD_train, augment=True)
train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, BATCH_SIZE = BATCH_SIZE,
                                             shuffle = True, num_workers = 8)

test_image_datasets = my_dataloader(image_dir_test, center_test, test_lefttop_pixel, test_rightbottom_pixel, keypointsUVD_test, augment=False)
test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, BATCH_SIZE = BATCH_SIZE,
                                             shuffle = False, num_workers = 8)




def train(net):
    post_process = A2JPostProcess(None, None, [IMG_HEIGHT//16, IMG_WIDTH//16], 16)
    criterion = A2JLoss(None, None, [IMG_HEIGHT//16,IMG_WIDTH//16], 16, SPATIAL_FACTOR)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, WEIGHT_DECAY=WEIGHT_DECAY)
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

        # Training loop
        for i, (img, label) in enumerate(train_dataloaders):

            img, label = img.cuda(), label.cuda()

            heads  = net(img)
            #print(regression)
            optimizer.zero_grad()

            Cls_loss, Reg_loss = criterion(heads, label)

            loss = 1*Cls_loss + Reg_loss*REG_LOSS_FACTOR
            loss.backward()
            optimizer.step()

            train_loss_add = train_loss_add + (loss.item())*len(img)
            Cls_loss_add = Cls_loss_add + (Cls_loss.item())*len(img)
            Reg_loss_add = Reg_loss_add + (Reg_loss.item())*len(img)

            # printing loss info
            if i%10 == 0:
                print('epoch: ',epoch, ' step: ', i, 'Cls_loss ',Cls_loss.item(), 'Reg_loss ',Reg_loss.item(), ' total loss ',loss.item())

        scheduler.step(epoch)


        train_loss_add = train_loss_add / NUM_FRAME_TRAIN
        Cls_loss_add = Cls_loss_add / NUM_FRAME_TRAIN
        Reg_loss_add = Reg_loss_add / NUM_FRAME_TRAIN
        print('mean train_loss_add of 1 sample: %f, #train_indexes = %d' %(train_loss_add, NUM_FRAME_TRAIN))
        print('mean Cls_loss_add of 1 sample: %f, #train_indexes = %d' %(Cls_loss_add, NUM_FRAME_TRAIN))
        print('mean Reg_loss_add of 1 sample: %f, #train_indexes = %d' %(Reg_loss_add, NUM_FRAME_TRAIN))

        Error_test = 0
        Error_train = 0
        Error_test_wrist = 0

        if (epoch % 1 == 0):
            net = net.eval()
            output = torch.FloatTensor()
            outputTrain = torch.FloatTensor()

            for i, (img, label) in tqdm(enumerate(test_dataloaders)):
                with torch.no_grad():
                    img, label = img.cuda(), label.cuda()
                    heads = net(img)
                    pred_keypoints = post_process(heads, voting=False)
                    output = torch.cat([output,pred_keypoints.data.cpu()], 0)

            result = output.cpu().data.numpy()
            Error_test = errorCompute(result,keypointsUVD_test, center_test)
            print('epoch: ', epoch, 'Test error:', Error_test)
            saveNamePrefix = '%s/net_%d_wetD_' % (save_dir, epoch) + str(WEIGHT_DECAY) + '_depFact_' + str(SPATIAL_FACTOR) + '_RegFact_' + str(REG_LOSS_FACTOR) + '_rndShft_' + str(RAND_SHIFT_CROP)
            torch.save(net.state_dict(), saveNamePrefix + '.pth')

        # log
        logging.info('Epoch#%d: total loss=%.4f, Cls_loss=%.4f, Reg_loss=%.4f, Err_test=%.4f, lr = %.6f'
        %(epoch, train_loss_add, Cls_loss_add, Reg_loss_add, Error_test, scheduler.get_lr()[0]))



def test():
    net = model.A2J_model(num_classes = NUM_KEYPOINT)
    net.load_state_dict(torch.load(model_dir))
    net = net.cuda()
    net.eval()

    post_process = anchor.post_process(shape=[IMG_HEIGHT//16,IMG_WIDTH//16],stride=16,P_h=None, P_w=None)

    output = torch.FloatTensor()
    for i, (img, label) in tqdm(enumerate(test_dataloaders)):
        with torch.no_grad():

            img, label = img.cuda(), label.cuda()
            heads = net(img)
            pred_keypoints = post_process(heads,voting=False)
            output = torch.cat([output,pred_keypoints.data.cpu()], 0)


    result = output.cpu().data.numpy()
    writeTxt(result, center_test)
    error = errorCompute(result, keypointsUVD_test, center_test)
    print('Error:', error)


def errorCompute(source, target, center):
    assert np.shape(source)==np.shape(target), "source has different shape with target"

    Test1_ = source.copy()
    target_ = target.copy()
    Test1_[:, :, 0] = source[:,:,1]
    Test1_[:, :, 1] = source[:,:,0]
    Test1 = Test1_  # [x, y, z]

    center_pixel = center.copy()
    centre_world = pixel2world(center.copy(), fx, fy, u0, v0)

    centerlefttop = centre_world.copy()
    centerlefttop[:,0,0] = centerlefttop[:,0,0]-THRESH_XY
    centerlefttop[:,0,1] = centerlefttop[:,0,1]+THRESH_XY

    centerrightbottom = centre_world.copy()
    centerrightbottom[:,0,0] = centerrightbottom[:,0,0]+THRESH_XY
    centerrightbottom[:,0,1] = centerrightbottom[:,0,1]-THRESH_XY

    lefttop_pixel = world2pixel(centerlefttop, fx, fy, u0, v0)
    rightbottom_pixel = world2pixel(centerrightbottom, fx, fy, u0, v0)

    for i in range(len(Test1_)):
        Xmin = max(lefttop_pixel[i,0,0], 0)
        Ymin = max(lefttop_pixel[i,0,1], 0)
        Xmax = min(rightbottom_pixel[i,0,0], 320*2 - 1)
        Ymax = min(rightbottom_pixel[i,0,1], 240*2 - 1)

        Test1[i,:,0] = Test1_[i,:,0]*(Xmax-Xmin)/IMG_WIDTH + Xmin  # x
        Test1[i,:,1] = Test1_[i,:,1]*(Ymax-Ymin)/IMG_HEIGHT + Ymin  # y
        Test1[i,:,2] = source[i,:,2] + center[i][0][2]

    labels = pixel2world(target_, fx, fy, u0, v0)
    outputs = pixel2world(Test1.copy(), fx, fy, u0, v0)

    errors = np.sqrt(np.sum((labels - outputs) ** 2, axis=2))

    return np.mean(errors)



def writeTxt(result, center):

    resultUVD_ = result.copy()
    resultUVD_[:, :, 0] = result[:,:,1]
    resultUVD_[:, :, 1] = result[:,:,0]
    resultUVD = resultUVD_  # [x, y, z]

    center_pixel = center.copy()
    centre_world = pixel2world(center.copy(), fx, fy, u0, v0)

    centerlefttop = centre_world.copy()
    centerlefttop[:,0,0] = centerlefttop[:,0,0]-THRESH_XY
    centerlefttop[:,0,1] = centerlefttop[:,0,1]+THRESH_XY

    centerrightbottom = centre_world.copy()
    centerrightbottom[:,0,0] = centerrightbottom[:,0,0]+THRESH_XY
    centerrightbottom[:,0,1] = centerrightbottom[:,0,1]-THRESH_XY

    lefttop_pixel = world2pixel(centerlefttop, fx, fy, u0, v0)
    rightbottom_pixel = world2pixel(centerrightbottom, fx, fy, u0, v0)


    for i in range(len(result)):
        Xmin = max(lefttop_pixel[i,0,0], 0)
        Ymin = max(lefttop_pixel[i,0,1], 0)
        Xmax = min(rightbottom_pixel[i,0,0], 320*2 - 1)
        Ymax = min(rightbottom_pixel[i,0,1], 240*2 - 1)

        resultUVD[i,:,0] = resultUVD_[i,:,0]*(Xmax-Xmin)/IMG_WIDTH + Xmin  # x
        resultUVD[i,:,1] = resultUVD_[i,:,1]*(Ymax-Ymin)/IMG_HEIGHT + Ymin  # y
        resultUVD[i,:,2] = result[i,:,2] + center[i][0][2]

    resultReshape = resultUVD.reshape(len(result), -1)

    with open(os.path.join(save_dir, result_file), 'w') as f:
        for i in range(len(resultReshape)):
            for j in range(NUM_KEYPOINT*3):
                f.write(str(resultReshape[i, j])+' ')
            f.write('\n')

    f.close()

if __name__ == '__main__':
    import torch.nn as nn

    from activations import h_sigmoid, h_swish
    from blocks import SqueezeExcite
    from models.a2j import MNV3Backbone, A2J

    config = [
        #k, s, ex, out, nl,                    se
        [3, 2, 16, 16,  nn.ReLU(inplace=True), SqueezeExcite(16)],
        [3, 2, 72, 24,  nn.ReLU(inplace=True), nn.Identity()],
        [3, 1, 88, 24,  nn.ReLU(inplace=True), nn.Identity()],
        [5, 2, 96, 40,  h_swish(), SqueezeExcite(40)],
        [5, 1, 240, 40, h_swish(), SqueezeExcite(40)],
        [5, 1, 240, 40, h_swish(), SqueezeExcite(40)],
        [5, 1, 120, 48, h_swish(), SqueezeExcite(48)],
        [5, 1, 144, 48, h_swish(), SqueezeExcite(48)],
        [5, 2, 288, 96, h_swish(), SqueezeExcite(96)],
        [5, 1, 576, 96, h_swish(), SqueezeExcite(96)],
        [5, 1, 576, 96, h_swish(), SqueezeExcite(96)]
    ]
    backbone = MNV3Backbone(config)
    a2j = A2J(backbone, num_classes=NUM_KEYPOINT)


    train(a2j)
    test(a2j)
