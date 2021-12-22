import numpy as np
import cv2
# from matplotlib import pyplot as plt

def sample_for_dct():
    y = cv2.imread('/home/loachex/DF_detection_stevenYan/image_140.jpg')
    print(y.shape)
    y_b = y[:, :, 0].astype(np.float32)
    y_g = y[:, :, 1].astype(np.float32)
    y_r = y[:, :, 2].astype(np.float32)

    Y_b = cv2.dct(y_b)
    Y_g = cv2.dct(y_g)
    Y_r = cv2.dct(y_r)

    print(y_b.shape)
    print(Y_b.shape)

    bgr_dct = np.concatenate((Y_b, Y_g, Y_r), axis=1).reshape(224, 224, 3)
    print(bgr_dct.shape)
    cv2.imshow("dct_bgr", bgr_dct)
    cv2.waitKey(0)

    # cv2.imshow("dct_b", Y_g)
    # cv2.waitKey(0)
    # for i in range(0,240):
    #      for j in range(0,320):
    #          if i > 100 or j > 100:
    #              Y[i,j] = 0
    # cv2.imshow("Dct",Y)
    # y2 = cv2.idct(Y)
    # # print(y2.dtype)
    # cv2.imshow("iDCT",y2.astype(np.uint8))
    # cv2.waitKey(0)

def convert_img_to_dct(np_image):
    y_b = np_image[:, :, 0].astype(np.float32)
    y_g = np_image[:, :, 1].astype(np.float32)
    y_r = np_image[:, :, 2].astype(np.float32)

    Y_b = cv2.dct(y_b)
    Y_g = cv2.dct(y_g)
    Y_r = cv2.dct(y_r)

    bgr_dct = np.concatenate((Y_b, Y_g, Y_r), axis=1).reshape(224, 224, 3)
    return bgr_dct