import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def visulize_attention_ratio(img_path, save_path, mask_index, attention_mask, ratio=0.5, cmap="jet", save_image=False,
                             save_original_image=False):
    """
    img_path:   image file path to load
    save_path:  image file path to save
    attention_mask: 2-D attention map with np.array type, e.g, (h, w) or (w, h)
    ratio:  scaling factor to scale the output h and w
    cmap:   attention style, default: "jet"
    """
    print("load image from: ", img_path)
    img = Image.open(img_path, mode='r')
    img_h, img_w = img.size[0], img.size[1]
    
    top = (img_h-224)//2
    left = (img_w-224)//2
    right = left + 224
    bottom = top + 224

    img = img.crop((left, top, right, bottom))
    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))
    # scale the image
    img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((img_h, img_w))
    plt.imshow(img, alpha=1)
    plt.axis('off')
    # print(attention_mask.shape)

    mask = cv2.resize(attention_mask, (img_h, img_w))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)

    if save_image:
        img_name = img_path.split('/')[-1].split('.')[0] + '_' + str(mask_index) + "_with_attention.jpg"
        img_with_attention_save_path = os.path.join(save_path, img_name)
        # pre-process before saving
        print("save image to: " + save_path + " as " + img_name)
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1,  left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(img_with_attention_save_path, dpi=300)
        

    if save_original_image:
        print("save original image at the same time")
        img_name = img_path.split('/')[-1].split('.')[0] + "_original.jpg"
        original_image_save_path = os.path.join(save_path, img_name)
        img.save(original_image_save_path, quality=100)

if __name__ == "__main__":
    # CDULZ 
    att_path = "visualize/charades/attn_map/attn.pkl"
    img_path = "data/charades/Charades_rgb/N93LD/"
    save_path = "visualize/charades/img_with_attn"
    d = torch.load(att_path)
    map_cs = []
    map_ct = []
    for i, item in enumerate(d):
        item = item.cpu().numpy()
        if i % 2 != 0:
            map_ct.append(item)
        else:
            # print(item.shape)
            item = np.reshape(item, (10*157, 7, 7))
            map_cs.append(item)
    print(len(map_cs))
    # map_cc_t = d['cc_t']
    # map_cc_s = d['cc_s']
    # map_cs = d['cs']
    # map_ct = d['ct']
    # map_cs = np.reshape(map_cs, (30,157,8,8))
    # map_cs = map_cs[:,0,:]
    # map_cs = list(map_cs[:])
    img_list = os.listdir(img_path)
    for img in img_list[::10]:
        for i, att_mask in enumerate(map_cs):
            j = 0
            length = att_mask.shape[0]
            if not os.path.exists(save_path+"/"+str(i)):
                os.mkdir(save_path+"/"+str(i))
            # for j in range(0, length, 10):
            visulize_attention_ratio(img_path+img, save_path+"/"+str(i), j, att_mask[j], save_image=True)