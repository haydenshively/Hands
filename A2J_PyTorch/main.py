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


    if '--pretrained' in sys.argv:
        saved = torch.load('mbv3_small.pth.tar', map_location=torch.device('cpu'))
        state_dict = OrderedDict()
        for key in saved['state_dict']:
            if key.startswith('module.'):
                state_dict[key[7:]] = saved['state_dict'][key]
            else:
                state_dict[key] = saved['state_dict'][key]
        backbone.load_state_dict(state_dict)

        for param in backbone.parameters():
            param.requires_grad = False


    a2j = A2J(backbone, num_classes=NUM_KEYPOINT)
    use_gpu = '--gpu' in sys.argv
    if use_gpu:
        a2j = a2j.cuda()
    if '--train' in sys.argv:
        train(a2j, use_gpu)
    if '--test' in sys.argv:
        test(a2j, use_gpu)
