import os, sys
import time
import torch

current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current)

@torch.no_grad()
@torch.autocast('cuda')
def vgg_pyramide_loss(source_image, dest_image, vgg, pyramid, scales):
    loss_weights = {
        'generator_gan': 1,                  
        'discriminator_gan': 1,
        'feature_matching': [10, 10, 10, 10],
        'perceptual': [10, 6, 6, 4, 1],
        'equivariance_value': 10,
        'equivariance_jacobian': 0,
        'keypoint': 10,
        'headpose': 20,
        'expression': 5
    }

    pyramid_source = pyramid(source_image)
    pyramid_dest = pyramid(dest_image)

    value_total = 0
    restricted_scales = (int(0.3*len(scales)), int(0.7*len(scales)))
    for index in range(len(scales)):
        if index >= restricted_scales[0] and index <= restricted_scales[1]:
            st = time.time()
            x_vgg = vgg(pyramid_source['prediction_' + str(scales[index])])
            y_vgg = vgg(pyramid_dest['prediction_' + str(scales[index])])

            for i, _ in enumerate(loss_weights['perceptual']):
                value = torch.abs(x_vgg[i].detach() - y_vgg[i].detach()).mean(dim=(1, 2, 3))
                value_total += loss_weights['perceptual'][i] * value

    return value_total.tolist()
