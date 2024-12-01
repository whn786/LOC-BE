import pdb
import torch
import torch.nn.functional as F

def cam_to_label(cam, cls_label, img_box=None, bkg_thre=None, high_thre=None, low_thre=None, ignore_mid=False, ignore_index=None):
    b, c, h, w = cam.shape
    #pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
    valid_cam = cls_label_rep * cam
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)
    _pseudo_label += 1
    _pseudo_label[cam_value<=bkg_thre] = 0

    if img_box is None:
        return _pseudo_label

    if ignore_mid:
        _pseudo_label[cam_value<=high_thre] = ignore_index
        _pseudo_label[cam_value<=low_thre] = 0
    pseudo_label = torch.ones_like(_pseudo_label) * ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]]

    return valid_cam, pseudo_label

def cam_to_roi_mask2(cam, cls_label, hig_thre=None, low_thre=None):
    b, c, h, w = cam.shape
    #pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
    valid_cam = cls_label_rep * cam
    cam_value, _ = valid_cam.max(dim=1, keepdim=False)
    # _pseudo_label += 1
    roi_mask = torch.ones_like(cam_value, dtype=torch.int16)
    roi_mask[cam_value<=low_thre] = 0
    roi_mask[cam_value>=hig_thre] = 2

    return roi_mask

def get_valid_cam(cam, cls_label):
    b, c, h, w = cam.shape
    #pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
    valid_cam = cls_label_rep * cam

    return valid_cam

def ignore_img_box(label, img_box, ignore_index):

    pseudo_label = torch.ones_like(label) * ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = label[idx, coord[0]:coord[1], coord[2]:coord[3]]

    return pseudo_label

def crop_from_roi_neg(images, roi_mask=None, crop_num=8, crop_size=96):

    crops = []
    
    b, c, h, w = images.shape

    temp_crops = torch.zeros(size=(b, crop_num, c, crop_size, crop_size)).to(images.device)
    flags = torch.ones(size=(b, crop_num+2)).to(images.device)
    margin = crop_size//2

    for i1 in range(b):
        roi_index = (roi_mask[i1, margin:(h-margin), margin:(w-margin)] <= 1).nonzero()
        if roi_index.shape[0] < crop_num:
            roi_index = (roi_mask[i1, margin:(h-margin), margin:(w-margin)] >= 0).nonzero() ## if NULL then random crop
        rand_index = torch.randperm(roi_index.shape[0])
        crop_index = roi_index[rand_index[:crop_num], :]
        
        for i2 in range(crop_num):
            h0, w0 = crop_index[i2, 0], crop_index[i2, 1] # centered at (h0, w0)
            temp_crops[i1, i2, ...] = images[i1, :, h0:(h0+crop_size), w0:(w0+crop_size)]
            temp_mask = roi_mask[i1, h0:(h0+crop_size), w0:(w0+crop_size)]
            if temp_mask.sum() / (crop_size*crop_size) <= 0.2:
                ## if ratio of uncertain regions < 0.2 then negative
                flags[i1, i2+2] = 0
    
    _crops = torch.chunk(temp_crops, chunks=crop_num, dim=1,)
    crops = [c[:, 0] for c in _crops]

    return crops, flags

def multi_scale_cam2(model, inputs, scales):
    '''process cam and aux-cam'''
    # cam_list, tscam_list = [], []
    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
        # print('inputs_cat:',inputs_cat.shape)
        _cam_aux, _cam = model(inputs_cat, cam_only=True)

        _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
        _cam_aux = F.interpolate(_cam_aux, size=(h,w), mode='bilinear', align_corners=False)
        _cam_aux = torch.max(_cam_aux[:b,...], _cam_aux[b:,...].flip(-1))

        cam_list = [F.relu(_cam)]
        cam_aux_list = [F.relu(_cam_aux)]

        for s in scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, size=(int(s*h), int(s*w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _cam_aux, _cam = model(inputs_cat, cam_only=True)

                _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
                _cam_aux = F.interpolate(_cam_aux, size=(h,w), mode='bilinear', align_corners=False)
                _cam_aux = torch.max(_cam_aux[:b,...], _cam_aux[b:,...].flip(-1))

                cam_list.append(F.relu(_cam))
                cam_aux_list.append(F.relu(_cam_aux))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5

        cam_aux = torch.sum(torch.stack(cam_aux_list, dim=0), dim=0)
        cam_aux = cam_aux + F.adaptive_max_pool2d(-cam_aux, (1, 1))
        cam_aux /= F.adaptive_max_pool2d(cam_aux, (1, 1)) + 1e-5

    return cam, cam_aux
def multi_scale_cam22(model, inputs, scales):
    '''process cam '''
    # cam_list, tscam_list = [], []
    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
        print(inputs_cat.shape)
        _cam = model(inputs_cat, cam_only=True,need_seg=False,need_aux=False)
        
        _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
        

        cam_list = [F.relu(_cam)]
       

        for s in scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, size=(int(s*h), int(s*w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _cam = model(inputs_cat, cam_only=True,need_seg=False,need_aux=False)

                _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
                

                cam_list.append(F.relu(_cam))
                

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5

     

    return cam
def label_to_aff_mask(cam_label, ignore_index=255):
    
    b,h,w = cam_label.shape

    _cam_label = cam_label.reshape(b, 1, -1)
    _cam_label_rep = _cam_label.repeat([1, _cam_label.shape[-1], 1])
    _cam_label_rep_t = _cam_label_rep.permute(0,2,1)
    aff_label = (_cam_label_rep == _cam_label_rep_t).type(torch.long)
    
    for i in range(b):
        aff_label[i, :, _cam_label_rep[i, 0, :]==ignore_index] = ignore_index
        aff_label[i, _cam_label_rep[i, 0, :]==ignore_index, :] = ignore_index
    aff_label[:, range(h*w), range(h*w)] = ignore_index
    return aff_label


def refine_cams_with_bkg_v2(ref_mod=None, images=None, cams=None, cls_labels=None, high_thre=None, low_thre=None, ignore_index=False,  img_box=None, down_scale=2):

    b,_,h,w = images.shape
    _images = F.interpolate(images, size=[h//down_scale, w//down_scale], mode="bilinear", align_corners=False)

    bkg_h = torch.ones(size=(b,1,h,w))*high_thre
    bkg_h = bkg_h.to(cams.device)
    bkg_l = torch.ones(size=(b,1,h,w))*low_thre
    bkg_l = bkg_l.to(cams.device)

    bkg_cls = torch.ones(size=(b,1))
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    refined_label = torch.ones(size=(b, h, w)) * ignore_index
    refined_label = refined_label.to(cams.device)
    refined_label_h = refined_label.clone()
    refined_label_l = refined_label.clone()
    
    cams_with_bkg_h = torch.cat((bkg_h, cams), dim=1)
    _cams_with_bkg_h = F.interpolate(cams_with_bkg_h, size=[h//down_scale, w//down_scale], mode="bilinear", align_corners=False)#.softmax(dim=1)
    cams_with_bkg_l = torch.cat((bkg_l, cams), dim=1)
    _cams_with_bkg_l = F.interpolate(cams_with_bkg_l, size=[h//down_scale, w//down_scale], mode="bilinear", align_corners=False)#.softmax(dim=1)

    for idx, coord in enumerate(img_box):

        valid_key = torch.nonzero(cls_labels[idx,...])[:,0]
        valid_cams_h = _cams_with_bkg_h[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)
        valid_cams_l = _cams_with_bkg_l[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)

        _refined_label_h = _refine_cams(ref_mod=ref_mod, images=_images[[idx],...], cams=valid_cams_h, valid_key=valid_key, orig_size=(h, w))
        _refined_label_l = _refine_cams(ref_mod=ref_mod, images=_images[[idx],...], cams=valid_cams_l, valid_key=valid_key, orig_size=(h, w))
        
        refined_label_h[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_h[0, coord[0]:coord[1], coord[2]:coord[3]]
        refined_label_l[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_l[0, coord[0]:coord[1], coord[2]:coord[3]]

    refined_label = refined_label_h.clone()
    refined_label[refined_label_h == 0] = ignore_index
    refined_label[(refined_label_h + refined_label_l) == 0] = 0

    return refined_label

def _refine_cams(ref_mod, images, cams, valid_key, orig_size):

    refined_cams = ref_mod(images, cams)
    refined_cams = F.interpolate(refined_cams, size=orig_size, mode="bilinear", align_corners=False)
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label

def blur_region(orig_img, roi_mask, kernel_size=21, sigma=7):
    blur_mask = ((roi_mask == 2)|(roi_mask == 0))


    kernel = torch.exp(-(torch.arange(kernel_size).float() - kernel_size // 2) ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1).to(orig_img.device)

    resized = F.interpolate(orig_img, size=(blur_mask.shape[1], blur_mask.shape[2]), mode='bilinear', align_corners=False)
    blurred = F.conv2d(resized, kernel, padding=kernel_size // 2, groups=3)

    blurred_resized = F.interpolate(blurred, size=orig_img.shape[2:], mode='bilinear', align_corners=False)
    blurred_mask = blurred_resized * blur_mask.float().unsqueeze(1)
    
    inverse_mask = torch.logical_not(blur_mask).float().unsqueeze(1)
    orig_img_masked = orig_img.float() * inverse_mask

    output = orig_img_masked + blurred_mask
    output = torch.clamp(output, 0, 1)
    return output


def merge_cam_s(cam1, cam2, weight):
    """
    线性地融合两个CAM特征图。

    参数:
      cam1 (np.array): 第一个CAM特征图。
      cam2 (np.array): 第二个CAM特征图。
      weight (float): 权重，范围0到1。

    返回:
      np.array: 融合后的特征图。
    """
    assert 0 <= weight <= 1, '权重值应位于0和1之间。'
    assert cam1.shape == cam2.shape, '两个特征图的大小应相同。'

    cam_fused = cam1 * weight + cam2 * (1 - weight)
    return cam_fused


def mask(inputs,roi_mask,BQD=1):
    # Create a mask for the regions to erase
    if BQD==1:
        erase_mask =((roi_mask == 2)|(roi_mask == 0))
    else :
        erase_mask =((roi_mask == 2)|(roi_mask == 1))
    # Erase the regions from the input tensor
    inputs2 = inputs.clone()  # Make a copy of the input tensor
    erase_mask = erase_mask.unsqueeze(1).repeat(1, 3, 1, 1)
    inputs2[erase_mask] = 0  # Set the values of the regions to 0
    return inputs2


def cams_to_refine_label(cam_label, mask=None, ignore_index=255):
    
    b,h,w = cam_label.shape

    cam_label_resized = F.interpolate(cam_label.unsqueeze(1).type(torch.float32), size=[h//16, w//16], mode="nearest")

    _cam_label = cam_label_resized.reshape(b, 1, -1)
    _cam_label_rep = _cam_label.repeat([1, _cam_label.shape[-1], 1])
    _cam_label_rep_t = _cam_label_rep.permute(0,2,1)
    ref_label = (_cam_label_rep == _cam_label_rep_t).type(torch.long)
    #ref_label[(_cam_label_rep+_cam_label_rep_t) == 0] = ignore_index
    for i in range(b):

        if mask is not None:
            ref_label[i, mask==0] = ignore_index

        ref_label[i, :, _cam_label_rep[i, 0, :]==ignore_index] = ignore_index
        ref_label[i, _cam_label_rep[i, 0, :]==ignore_index, :] = ignore_index

    return ref_label


import torch
import torchvision.transforms.functional as TF

def random_crop(img_tensor: torch.Tensor, roimask_tensor: torch.Tensor, crop_size: int, num_crops: int) -> torch.Tensor:
    batch_size, _, height, width = img_tensor.shape
    
    assert height == roimask_tensor.shape[1] and width == roimask_tensor.shape[2], "The sizes of the input tensor and mask tensor are inconsistent"
    assert crop_size <= height and crop_size <= width, "Crop size greater than the size of the input tensor"

    roi_indices = torch.nonzero(roimask_tensor == 2, as_tuple=False)
    
    num_rois = roi_indices.shape[0]
    if num_rois == 0:
        raise ValueError("There is no marker for a specific area in the mask")

    crop_samples = []
    for i in range(num_crops):
        index = torch.randint(0, num_rois, (1,))[0]
        h_center, w_center = roi_indices[index][1], roi_indices[index][2]
        
        h_crop_start = h_center - crop_size // 2
        w_crop_start = w_center - crop_size // 2
        h_crop_end = h_crop_start + crop_size
        w_crop_end = w_crop_start + crop_size

        h_crop_start = max(h_crop_start, 0)  
        w_crop_start = max(w_crop_start, 0) 
        h_crop_end = min(h_crop_end, height)  
        w_crop_end = min(w_crop_end, width)  

        # 只保留完全重叠的样本
        if h_crop_start <= h_center <= h_crop_end - 1 and w_crop_start <= w_center <= w_crop_end - 1:
            crop_window = img_tensor[:, :, h_crop_start:h_crop_end, w_crop_start:w_crop_end]
            # print(crop_window.shape)
            crop_samples.append(crop_window)

        if len(crop_samples) >= num_crops:
            break

    if len(crop_samples) == 0:
        
        return torch.empty((0,) + img_tensor.shape[1:], dtype=img_tensor.dtype, device=img_tensor.device)

    return crop_samples
import cv2
import numpy as np
def MSR(img, sigma_list):
    img_list = []
    for i in sigma_list:
        img_list.append(cv2.GaussianBlur(img, (0, 0), i))
    img_msr = np.zeros_like(img_list[0]).astype(np.float16)
    
    for i in range(len(sigma_list)):
        img_msr += np.log(img) - np.log(img_list[i])
    img_msr = np.exp(img_msr / len(sigma_list))
    return img_msr