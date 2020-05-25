if __name__ == '__main__':
    import sys
    from collections import OrderedDict

    import torch
    import torch.nn as nn

    from activations import h_sigmoid, h_swish
    from blocks import SqueezeExcite
    from models.a2j import MNV3Backbone, A2J
    from training.nyu import train, test, NUM_KEYPOINT

    config = [
        #k, s, ex, out, dil,   nl,                    se
        [3, 2, 16, 16,  1,      nn.ReLU(inplace=True), SqueezeExcite(16)],
        [3, 2, 72, 24,  1,      nn.ReLU(inplace=True), nn.Identity()],
        [3, 1, 88, 24,  1,      nn.ReLU(inplace=True), nn.Identity()],
        [5, 2, 96, 40,  1,      h_swish(), SqueezeExcite(40)],
        [5, 1, 240, 40, 1,      h_swish(), SqueezeExcite(40)],
        [5, 1, 240, 40, 1,      h_swish(), SqueezeExcite(40)],
        [5, 1, 120, 48, 1,      h_swish(), SqueezeExcite(48)],
        [5, 1, 144, 48, 1,      h_swish(), SqueezeExcite(48)],
        [5, 1, 288, 96, 1,      h_swish(), SqueezeExcite(96)],
        [5, 1, 288, 96, 1,      h_swish(), SqueezeExcite(96)],
        [5, 1, 288, 96, 1,      h_swish(), SqueezeExcite(96)]
    ]# NOTE: A2J paper says dilation of last 2 layers should be 2


    backbone = MNV3Backbone(config, index_C4=9)

    is_3D = '--3d' in sys.argv
    a2j = A2J(backbone, num_classes=NUM_KEYPOINT, is_3D=is_3D)
    use_gpu = '--gpu' in sys.argv
    if use_gpu:
        a2j = a2j.cuda()


    if '--train' in sys.argv:
        train(a2j, use_gpu, is_3D)
    if '--test' in sys.argv:
        for pth_path in sys.argv[::-1]:
            if pth_path == '--test': break
            test(a2j, pth_path, use_gpu, is_3D)
