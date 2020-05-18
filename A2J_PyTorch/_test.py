if __name__ == '__main__':
    from training import nyu

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
    a2j = A2J(backbone, num_classes=15)
    # print(a2j)

    print('Total params: %.2fM' % (sum(p.numel() for p in a2j.parameters())/1000000.0))


if __name__ == '__main__':
    import torch
    from torchvision import transforms
    from collections import OrderedDict
    from PIL import Image

    from models.mobilenetv3 import mobilenetv3_small

    net = mobilenetv3_small()
    # from torchsummary import summary
    # summary(net, (3, 256, 256))
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))

    saved = torch.load('mbv3_small.pth.tar', map_location=torch.device('cpu'))
    state_dict = OrderedDict()
    for key in saved['state_dict']:
        if key.startswith('module.'):
            state_dict[key[7:]] = saved['state_dict'][key]
        else:
            state_dict[key] = saved['state_dict'][key]

    net.load_state_dict(state_dict)

    loader = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

    image = Image.open('/Users/haydenshively/Desktop/watersnake.jpg')
    image = loader(image).float()
    image = image.unsqueeze(0)

    net.eval()
    res = net(image).data.numpy()[0]
    a = res.argmax()
    res[a] = -1
    b = res.argmax()
    res[b] = -1
    c = res.argmax()
    res[c] = -1
    d = res.argmax()

    print((a,b,c,d))
