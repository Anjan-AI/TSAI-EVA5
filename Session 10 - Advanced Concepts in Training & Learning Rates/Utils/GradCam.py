# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 19:17:16 2020

@author: Anjan 
"""
import torch
import torch.nn.functional as F
class GradCAM:
    """Calculate GradCAM salinecy map.
    Args:
        input: input image with shape of (1, 3, H, W)
        class_idx (int): class index for calculating GradCAM.
                If not specified, the class index that makes the highest model prediction score will be used.
    Return:
        mask: saliency map of the same spatial dimension with input
        logit: model output
    A simple example:
        # initialize a model, model_dict and gradcam
        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)
        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)
        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)
    """

    def __init__(self, model, layer_name):
        self.model = model
        # self.layer_name = layer_name
        self.target_layer = layer_name

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]

        def forward_hook(module, input, output):
            self.activations['value'] = output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def saliency_map_size(self, *input_size):
        device = next(self.model.parameters()).device
        self.model(torch.zeros(1, 3, *input_size, device=device))
        return self.activations['value'].shape[2:]

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()

        logit = self.model(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
        
        self.gradients.clear()
        self.activations.clear()
        return saliency_map, logit
    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)





# ------------------------------------VISUALIZE_GRADCAM-------------------------------------------------------------

import cv2
def visualize_cam(mask, img, alpha=1.0):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]
    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b]) * alpha

    result = heatmap+img.cpu()
    result = result.div(result.max()).squeeze()

    return heatmap, result

#-------------------------------------------GradCam View (Initialisation)--------------------------------------------

from torchvision.utils import make_grid, save_image
import numpy as np
import matplotlib.pyplot as plt


def GradCamView(miscalssified_images,model,classes,layers,Figsize = (23,30),subplotx1 = 9, subplotx2 = 3):

    fig = plt.figure(figsize=Figsize)
    for i,k in enumerate(miscalssified_images):
        images1 = [miscalssified_images[i][0].cpu()/2+0.5]
        images2 =  [miscalssified_images[i][0].cpu()/2+0.5]
        for j in layers:
                g = GradCAM(model,j)
                mask, _= g(miscalssified_images[i][0].clone().unsqueeze_(0))
                heatmap, result = visualize_cam(mask,miscalssified_images[i][0].clone().unsqueeze_(0)/2+0.5 )
                images1.extend([heatmap])
                images2.extend([result])
        # Ploting the images one by one
        grid_image = make_grid(images1+images2,nrow=len(layers)+1,pad_value=1)
        npimg = grid_image.numpy()
        sub = fig.add_subplot(subplotx1, subplotx2, i+1) 
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        sub.set_title('P = '+classes[int(miscalssified_images[i][1])]+" A = "+classes[int(miscalssified_images[i][2])],fontweight="bold",fontsize=18)
        sub.axis("off")
        plt.tight_layout()
        fig.subplots_adjust(wspace=0)