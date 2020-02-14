from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import os
import pickle
import numpy as np
import cv2
import natsort
os.environ['CUDA_VISIBLE_DEVICES']='2'
config_file = '/home/keze/Codes/GTA5/configs/cascade_rcnn_hrnetv2p_w32_20e.py'
checkpoint_file = '/home/keze/Codes/GTA5/checkpoints/cascade_mask_rcnn_hrnetv2p_w32_20e_20190810-76f61cd0.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')

folderPath = "/home/keze/GTAV/rgbImgs_first"
dstFolderPath = "/home/keze/GTAV/drawn_rgbImgs_first_hret/"
pklFolderPath = "/home/keze/GTAV/first_result_pkls_hret/"

if not os.path.exists(pklFolderPath):
   os.mkdir(pklFolderPath)

if not os.path.exists(dstFolderPath):
    os.mkdir(dstFolderPath)

filelist = os.listdir(folderPath)
filelist = natsort.natsorted(filelist)

imgPathlist = list()
for filename in filelist:
    imgPathlist.append( os.path.join( folderPath, filename ) )
    # img = cv2.imread( os.path.join( folderPath, filename ) )
    # with open(folderPath + '/'+ filename, 'rb') as fid:
    #     img = pickle.load(fid)
    # try:
    #     img.shape;
    #     imgPathlist.append( os.path.join( folderPath, filename ) )
    # except:
    #     print(os.path.join( folderPath, filename ))
    
resultList = list()
for i, result in enumerate(inference_detector(model, imgPathlist)):
    resultList.append( (imgPathlist[i], result) )

    with open(pklFolderPath + str(i).zfill(9) + '_result.pkl', 'wb') as fid:
        pickle.dump( result, fid, pickle.HIGHEST_PROTOCOL )

    try:
        show_result(imgPathlist[i], result, model.CLASSES, out_file= dstFolderPath + 'result_{}.jpg'.format(i))
    except:
        print(imgPathlist[i])



# print(img.shape)
# result = inference_detector(model, img)
# print(result.shape)
# print(result[0, :])
# print(result[1, :])
# print(result[2, :])
# print(result[1999, :])
# print(model.CLASSES)

# show_result(img, result, model.CLASSES, out_file='predictions/result.jpg')


