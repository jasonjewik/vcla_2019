import os.path as osp
import argparse
import pickle
from color import color_val

import cv2
import time
import numpy as np
#from mmcv.parallel import DataContainer as DC
import mmcv
from mmdet.datasets.utils import to_tensor
from sklearn.metrics.pairwise import cosine_similarity
from mmdet.datasets.transforms import ImageTransform

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--start', type= int, help='starting index')
    parser.add_argument('--end', type= int, help = 'ending index')
    parser.add_argument('--gpu', type=str, choices=['0','1','2','3'], help='GPU')

    args = parser.parse_args()
    return args


def showFrameTime(previousTime):
    print('Total time used ' + str(time.time() - previousTime) + ' seconds.')
    print()

def preprocessInput(image, MODEL_INPUT_SIZE=(320, 160)):
    newImage = cv2.resize(image, MODEL_INPUT_SIZE)
    newImage = np.expand_dims(newImage, axis=0)
    newImage = newImage / 255

    return newImage

def postProcessOutput(image, MODEL_INPUT_SIZE=(320, 160), NEW_SIZE=(800, 450)):
    #Bring image from 0-1 to 0-255 and remove last axis
    newImage = image * 255
    newImage = newImage.astype(np.uint8)
    newImage = np.squeeze(newImage, axis=2)

    #Create blank image and add image to the GREEN channel
    overlay = np.zeros((MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0], 3))
    overlay[:, :, 1] = newImage
    overlay = overlay.astype(np.uint8)
    overlay = cv2.resize(overlay, NEW_SIZE, interpolation=cv2.INTER_NEAREST)

    return overlay

def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      thickness=1,
                      font_scale=0.5,
                      show=True,
                      win_name='',
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5

    if isinstance(img, str):
        if img.endswith(".pkl"):
            with open(img, 'rb') as f:
                img = pickle.load(f)
        else:
            img = cv2.imread(img)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, bbox_color, thickness=thickness)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        if len(bbox) > 4:
            if bbox[-1] > 1:
                label_text += '|ID: ' + str(int(bbox[-1]))
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

    if show:
        cv2.imshow(win_name, img)
        cv2.waitKey(wait_time)
    if out_file is not None:
        cv2.imwrite(out_file, img)

def show_result(img, result, class_names, out_file=None):
    bboxes = np.vstack(result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)
    imshow_det_bboxes(
        img,
        bboxes,
        labels,
        class_names= class_names,
        score_thr=0.3,
        out_file=out_file,
        wait_time = 1,
        )

def center_of_bbox(bbox):
    return [(bbox[0]+bbox[2])/2, (bbox[1] + bbox[3])/2]

def calculate_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def combine_bboxes(bb1,bb2):
    bbox = np.empty((1, 5), dtype='float32')

    bbox[0,0] = np.min((bb1[0], bb2[0]))
    bbox[0,1] = np.max((bb1[1], bb2[1]))
    bbox[0,2] = np.min((bb1[2], bb2[2]))
    bbox[0,3] = np.max((bb1[3], bb2[3]))
    bbox[0,4] = np.max((bb1[4], bb2[4]))
    return bbox


# def get_direction(bb1,bb2):
#     bb1_x, bb1_y = (bb1[0]+bb1[2])/2, (bb1[1]+bb1[3])/2
#     bb2_x, bb2_y = (bb2[0]+bb2[2])/2, (bb2[1]+bb2[3])/2
#     return np.array([bb2_x-bb1_x, bb2_y-bb1_y])


# def get_distance(bb1,bb2):
#     bb1_x, bb1_y = (bb1[0]+bb1[2])/2, (bb1[1]+bb1[3])/2
#     bb2_x, bb2_y = (bb2[0]+bb2[2])/2, (bb2[1]+bb2[3])/2
#     diff_x = bb2_x - bb1_x
#     diff_y = bb2_y - bb1_y
#     return math.sqrt(diff_x * diff_x + diff_y * diff_y)


def get_bboxes_and_labels_from_result(result):
    """
    Get {label: bboxes} dict from result
    Filter bouding bboxes with low confidence(<0.3)
    Combine bboxes that overlapped with each other(IoU>0.5)
    """
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    # Filter bouding bboxes with low confidence(<0.3)
    for i, bboxes in enumerate(bbox_result):
        if bboxes.size !=0:
            bboxes = bboxes[bboxes[:,4]> 0.3 ]
            bbox_result[i] = bboxes
    # Combine bboxes that overlapped with each other(IoU>0.5)
    for i, bboxes in enumerate(bbox_result):
        if bboxes.size !=0:
            bboxes_res = []
            for k in range(len(bboxes)):
                isMerged = False
                bbox = bboxes[k]
                for bbox_i in bboxes[k+1:]:
                    if calculate_iou(bbox, bbox_i) > 0.9:
                        bboxes_res.append(combine_bboxes(bbox.astype('float32'), bbox_i.astype('float32')))
                        isMerged = True
                        break

                if not isMerged:
                    bbox = np.reshape( bbox, (1, 5) )
                    bboxes_res.append(bbox)
            merged_bbox_res = None
            for bbox in bboxes_res:
                if merged_bbox_res is None:
                    merged_bbox_res = bbox
                else:
                    merged_bbox_res = np.concatenate( (merged_bbox_res, bbox), axis = 0 )
            if merged_bbox_res is not None:
                bbox_result[i] = merged_bbox_res
    # print(bbox_result)
    return bbox_result

def spatialPooling(featmap, grid_h, grid_w):
    assert featmap.shape[2] >= grid_h and featmap.shape[3] >= grid_w
    if grid_h == 1 and grid_w == 1:
       return np.mean(np.mean(featmap, axis=2), axis=2)
    else:
       print('NOT IMPLEMENTED!')


    #pooled_map = np.empty((grid_h, grid_w), dtype='float32')
    #splits_h = int(featmap.shape[0] / grid_h)
    #splits_w = int(featmap.shape[1] / grid_w)

    #for i in range(grid_h-1):
    #    for j in range(grid_w-1):
    #        pooled_map[i, j] = np.max(featmap[i * splits_h:(i+1) * splits_h, j*splits_w:(j+1)*splits_w])
    #
    ## print(splits_h, splits_w)
    ## for i in range(grid_h):
    #    # for j in range(grid_w):
    #for i in range(grid_h-1, grid_h):
    #    for j in range(grid_h-1, grid_w):
    #        if (i+2) * splits_h > featmap.shape[0] and (j+2) * splits_w > featmap.shape[1] :
    #            pooled_map[i, j] = np.max(featmap[i * splits_h:, j*splits_w:])
    #        elif (i+2) * splits_h > featmap.shape[0]:
    #            pooled_map[i, j] = np.max(featmap[i * splits_h:, j*splits_w:(j+1)*splits_w])
    #        elif (j+2) * splits_w > featmap.shape[1] > featmap.shape[1]:
    #            pooled_map[i, j] = np.max(featmap[i * splits_h:(i+1) * splits_h, j*splits_w:])
    #        else:
    #            pooled_map[i, j] = np.max(featmap[i * splits_h:(i+1) * splits_h, j*splits_w:(j+1)*splits_w])
    #return pooled_map


def calcCos(data1, data2, grid_h=1, grid_w=1):
    assert data1.shape[0] == data2.shape[0] and data1.shape[1] == data2.shape[1]
    data1_pooled = spatialPooling(data1, grid_h, grid_w)
    data2_pooled = spatialPooling(data2, grid_h, grid_w)

    data1_pooled = np.reshape(data1_pooled, (1, -1))
    data2_pooled = np.reshape(data2_pooled, (1, -1))
    return cosine_similarity(data1_pooled, data2_pooled)[0][0]


def calcDist(data1, data2, grid_h=1, grid_w=1):
    assert data1.shape[0] == data2.shape[0] and data1.shape[1] == data2.shape[1]
    data1_pooled = spatialPooling(data1, grid_h, grid_w)
    data2_pooled = spatialPooling(data2, grid_h, grid_w)

    #for n in range(data1.shape[0]):
    #    for c in range(data1.shape[1]):
    #        data1_pooled[n, c, :, :] = spatialPooling(data1[n, c, :, :], grid_h, grid_w)
    #        data2_pooled[n, c, :, :] = spatialPooling(data2[n, c, :, :], grid_h, grid_w)

    return np.linalg.norm(data1_pooled - data2_pooled)

def impad_gpu(img_gpu, shape, device, pad_val=0):
    """Pad an image to a certain shape.

    Args:
        img (ndarray): Image to be padded.
        shape (tuple): Expected padding shape.
        pad_val (number or sequence): Values to be filled in padding areas.

    Returns:
        ndarray: The padded image.
    """
    if not isinstance(pad_val, (int, float)):
        assert len(pad_val) == img_gpu.size(-1)
#    if len(shape) < len(img.shape):
#        shape = shape + (img.shape[-1], )
#    assert len(shape) == len(img.shape)
#    for i in range(len(shape) - 1):
#        assert shape[i] >= img.shape[i]
        
    pad = np.empty((shape[0], shape[1], 3), dtype=np.float32)
    pad[...] = pad_val
    pad_gpu = torch.from_numpy(pad).to(device)
    pad_gpu[:img_gpu.size(0), :img_gpu.size(1), ...] = img_gpu
    return pad_gpu    

def impad_to_multiple_gpu(img_gpu, divisor, device, pad_val=0):
    """Pad an image to ensure each edge to be multiple to some number.

    Args:
        img (ndarray): Image to be padded.
        divisor (int): Padded image edges will be multiple to divisor.
        pad_val (number or sequence): Same as :func:`impad`.

    Returns:
        ndarray: The padded image.
    """
    pad_h = int(np.ceil(img_gpu.size(0) / divisor)) * divisor
    pad_w = int(np.ceil(img_gpu.size(1) / divisor)) * divisor
    pad_shape = (pad_h, pad_w)
    return impad_gpu(img_gpu, pad_shape, device, pad_val), pad_shape

def img_transform_gpu(img, img_transform, scale, flip=False, keep_ratio=True, device='cuda:0'):
    mean_gpu = torch.from_numpy(img_transform.mean).to(device)
    std_gpu = torch.from_numpy(img_transform.std).to(device)
    
    if keep_ratio:
        img, scale_factor = mmcv.imrescale(img, scale, return_scale=True)
    img_shape = img.shape
    img_gpu = torch.from_numpy(img.astype(np.float32)).to(device)
    img_gpu = ( img_gpu - mean_gpu ) / std_gpu
    
    if img_transform.size_divisor is not None:
        img_gpu, pad_shape = impad_to_multiple_gpu(img_gpu, img_transform.size_divisor, device)
    img_gpu = img_gpu.permute(2, 0, 1)
    return img_gpu, img_shape, pad_shape, scale_factor    
    
def prepare_data_gpu(img, img_transform, img_scale, keep_ratio, device='cuda:0'):
    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform_gpu(
        img,
        img_transform,
        scale=img_scale,
        keep_ratio=keep_ratio, device=device)
    img = img.unsqueeze(0)
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False)
    ]    
    return dict(img=[img], img_meta=[img_meta])

def prepare_data(img, img_transform, img_scale, keep_ratio):
    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(
        img,
        scale=img_scale,
        keep_ratio=keep_ratio)
#    img = to_tensor(img).to(device).unsqueeze(0)
    img = to_tensor(img).unsqueeze(0)
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False)
    ]
    return dict(img=[img], img_meta=[img_meta])


def prepare_img(img, dataset):
    def prepare_single(img, scale, flip, proposal=None):
        _img, img_shape, pad_shape, scale_factor = dataset.img_transform(
            img, scale, flip, keep_ratio=dataset.resize_keep_ratio)
        # print(_img)
        _img = to_tensor(_img)
        # print(_img)
        _img_meta = dict(
            ori_shape=(720, 1280, 3),
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)

        if proposal is not None:
            if proposal.shape[1] == 5:
                score = proposal[:, 4, None]
                proposal = proposal[:, :4]
            else:
                score = None
            _proposal = dataset.bbox_transform(proposal, img_shape,
                                            scale_factor, flip)
            _proposal = np.hstack(
                [_proposal, score]) if score is not None else _proposal
            _proposal = to_tensor(_proposal)
        else:
            _proposal = None
        return _img, _img_meta, _proposal

    imgs = []
    img_metas = []
    proposals = []

    for scale in dataset.img_scales:
        _img, _img_meta, _proposal = prepare_single(
            img, scale, False, None)

        imgs.append(_img)
#        img_metas.append(DC(_img_meta, cpu_only=True))
        img_metas.append(_img_meta)
        proposals.append(_proposal)
        if dataset.flip_ratio > 0:
            _img, _img_meta, _proposal = prepare_single(
                img, scale, True, None)
            imgs.append(_img)
#            img_metas.append(DC(_img_meta, cpu_only=True))
            img_metas.append(_img_meta)
            proposals.append(_proposal)
    data = dict(img=imgs, img_meta=[img_metas])
    # print(data)
    if dataset.proposals is not None:
        data['proposals'] = proposals
    return data

def get_feature(img, bbox, model, dataset, imgName, i, j, feature_folder):
    try:
        with open(osp.join(feature_folder, imgName +'_'+ str(i)+'_'+str(j)) + '.pkl', 'rb') as fid:
            feat = pickle.load(fid)
    except:
        sub_img = img[int(bbox[1]):int(bbox[3]), int(bbox[0]): int(bbox[2])]
        img_data = prepare_img(sub_img, dataset)
        img_data['img'][0] = torch.unsqueeze(img_data['img'][0], 0).cuda()
        feat = model.extract_feat(img_data['img'][0])
        feat = feat[4].cpu().detach().numpy()
        with open(osp.join(feature_folder, imgName +'_'+ str(i)+'_'+str(j)) + '.pkl', 'wb') as fid:
            pickle.dump(feat, fid, pickle.HIGHEST_PROTOCOL)
    return feat

def calculate_iou_and_dist(img, data, imgName, total_obj_dict, theta):
    imgNameId = int(imgName[:9])
    # total_obj_dict = {imgNameId: {id: data}}
    res = dict()
    max_overlap_val = dict()
    max_overlap_indx = dict()
    max_cos_val= dict()
    max_cos_indx = dict()
    for idx in range(imgNameId-3, imgNameId-7, -3):
        if idx < 0:
            break
        if idx not in total_obj_dict.keys():
            continue
        obj_dict = total_obj_dict[idx]
        for id in obj_dict.keys():
            obj_bbox = obj_dict[id][0]
            obj_bbox_x = (obj_bbox[0]+obj_bbox[2])/2
            obj_bbox_y = (obj_bbox[1]+obj_bbox[3])/2
            # detection window
            window = [(1+theta) * obj_bbox[0] - theta * obj_bbox_x,
                      (1+theta) * obj_bbox[1] - theta * obj_bbox_y ,
                      (1+theta) * obj_bbox[2] - theta * obj_bbox_x ,
                      (1+theta) * obj_bbox[3] - theta * obj_bbox_y]
            obj_feat = obj_dict[id][1]
            window = np.array(window)
            window = np.reshape(window, (1,4))
            # mmcv.imshow_bboxes(img, window)

            max_overlap_val[id] = 0
            max_overlap_indx[id] = -1

            data_indx = list()
            for i in range(len(data)):
                bbox = data[i][0]
                bbox_x = (bbox[0]+bbox[2])/2
                bbox_y = (bbox[1]+bbox[3])/2
                if bbox_x >= window[0, 0] and bbox_x <= window[0, 2] and bbox_y >= window[0, 1] and bbox_y <= window[0, 3]:
                   data_indx.append(i)

            # print(id, len(data_indx))
            if len(data_indx) == 0:
               res[id] = None
               continue

            cos_val_list = list()
            for i in data_indx:
                bbox = data[i][0]
                feat = data[i][1]
                iou = calculate_iou(bbox,obj_bbox)
                cos_val_list.append(0)#calcCos(feat, obj_feat))

                if max_overlap_val[id] < iou:
                    max_overlap_val[id] = iou
                    max_overlap_indx[id] = i

            max_cos_indx[id] = data_indx[0]
            max_cos_val[id] = cos_val_list[0]
            if len(cos_val_list) > 1:
                sum_cos_val = 0
                for cos_val in cos_val_list:
                    sum_cos_val += cos_val

                max_cos_val[id] = sum_cos_val / len(cos_val_list)
                for j in range(len(cos_val_list)):
                    if max_cos_val[id] < cos_val_list[j]:
                        max_cos_val[id] = cos_val_list[j]
                        max_cos_indx[id] = data_indx[j]

            res[id] = [max_overlap_val[id], max_overlap_indx[id], max_cos_val[id], max_cos_indx[id]]
    return res



def mark_id(img, data, imgName, total_obj_dict, obj_id, theta=1):
    imgNameId = int(imgName[:9])
    iou_dist_dict= calculate_iou_and_dist(img, data, imgName, total_obj_dict, theta)
    
    #print(imgNameId)
    obj_dict = dict()
    total_obj_dict[imgNameId] = obj_dict
    not_found= []
    for id, tup in iou_dist_dict.items():
        # print(tup)
        if tup is None:
            not_found.append(id)
            continue
        max_overlap_val, max_overlap_indx, max_cos_val, max_cos_indx = tup[0], tup[1], tup[2], tup[3]
        if max_overlap_val > 0.5:# and max_cos_val > 0.985:
            data[max_overlap_indx][0][4] = id
            obj_dict[id] = data[max_overlap_indx]
        elif max_overlap_val < 0.5 and max_overlap_val >= 0.3:# and max_cos_val > 0.98:
            data[max_cos_indx][0][4] = id
            obj_dict[id] = data[max_cos_indx]
#        elif max_cos_val > 0.99:
#            data[max_cos_indx][0][4] = id
#            obj_dict[id] = data[max_cos_indx]
        else:
            not_found.append(id)

#    print(not_found)
    for id in not_found:
        if id in obj_dict.keys():
            obj_dict.pop(id)
    not_found = []
    for data_i in data:
        bbox = data_i[0]
        if bbox[4] < 1:
            obj_dict[obj_id] = data_i
            bbox[4] = obj_id
            obj_id += 1
    return obj_id, total_obj_dict

# def cosine_of_vectors(v1, v2):
#     v1_times_v2 = v1[0]* v2[0] + v1[1] * v2[1]
#     v1_norm = math.sqrt(v1[0]*v1[0] + v1[1] * v1[1])
#     v2_norm = math.sqrt(v2[0]*v2[0] + v2[1] * v2[1])
#     return v1_times_v2 / (v1_norm * v2_norm)

# def check_side_of_obj_top(player_center, player_vec, obj_bbox):
#     # 1 means right, 0 means left, -1 means back
#     player_perpend = [player_vec[1], - player_vec[0]]
#     obj_center = center_of_bbox(obj_bbox)
#     obj_vec = [obj_center[0]-player_center[0], obj_center[1] - player_center[1]]
#     if cosine_of_vectors(obj_vec, player_vec) < 0 :
#         return -1
#     if cosine_of_vectors(obj_vec, player_perpend) > 0:
#         return 1
#     else:
#         return 0

def check_side_of_obj_first(player_center, obj_bbox):
    obj_center = center_of_bbox(obj_bbox)
    if obj_center[0] > player_center[0]:
        return 1
    else:
        return 0

# def fetch_result(idx, feature_folder):
#     feature_img_folder = osp.join(feature_folder, str(idx).zfill(9))
#     feature_pkls = os.listdir(feature_img_folder)
#     data_list = list()
#     for pkl in feature_pkls:
#         with open(osp.join(feature_img_folder, pkl), 'rb') as f:
#             data = pickle.load(f)
#             data_list.append(data)
#     return data_list

# def show_result(img, result, class_names, out_file=None):
#     bboxes = np.vstack(result)
#     labels = [
#         np.full(bbox.shape[0], i, dtype=np.int32)
#         for i, bbox in enumerate(result)
#     ]
#     labels = np.concatenate(labels)
#     mmcv.imshow_det_bboxes(
#         img,
#         bboxes,
#         labels,
#         class_names=class_names,
#         score_thr=0.2,
#         out_file=out_file,
#         wait_time = 0
#         )
