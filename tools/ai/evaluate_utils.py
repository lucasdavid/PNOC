import cv2
import numpy as np

from tools.general.json_utils import read_json


def calculate_for_tags(pred_tags, gt_tags):
    """This function calculates precision, recall, and f1-score using tags.

    Args:
        pred_tags: 
            The type of variable is list.
            The type of each element is string.

        gt_tags:
            The type of variable is list.
            the type of each element is string.

    Returns:
        precision:
            pass

        recall:
            pass

        f1-score:
            pass
    """
    if len(pred_tags) == 0 and len(gt_tags) == 0:
        return 100, 100, 100
    elif len(pred_tags) == 0 or len(gt_tags) == 0:
        return 0, 0, 0
    
    pred_tags = np.asarray(pred_tags)
    gt_tags = np.asarray(gt_tags)

    precision = pred_tags[:, np.newaxis] == gt_tags[np.newaxis, :]
    recall = gt_tags[:, np.newaxis] == pred_tags[np.newaxis, :]
    
    precision = np.sum(precision) / len(precision) * 100
    recall = np.sum(recall) / len(recall) * 100
    
    if precision == 0 and recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * ((precision * recall) / (precision + recall))

    return precision, recall, f1_score


def calculate_mIoU(pred_mask, gt_mask):
    """This function is to calculate precision, recall, and f1-score using tags.

    Args:
        pred_mask: 
            The type of variable is numpy array.

        gt_mask:
            The type of variable is numpy array.

    Returns:
        miou:
            miou is meanIU.
    """
    inter = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    
    epsilon = 1e-5
    miou = (np.sum(inter) + epsilon) / (np.sum(union) + epsilon)
    return miou * 100


def accumulate_batch_iou_lowres(masks, cams, meters):
    for b in range(len(cams)):
        # c, h, w -> h, w, c
        cam = cams[b]
        gt_mask = masks[b]

        h, w, c = cam.shape
        gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        for th, meter in meters.items():
            bg = np.ones_like(cam[:, :, 0]) * th
            pred_mask = np.argmax(np.concatenate([bg[..., np.newaxis], cam], axis=-1), axis=-1)
            meter.add(pred_mask, gt_mask)


def result_miou_from_thresholds(iou_meters, classes):
    th_ = iou_ = None
    # miou_fg_ = None
    miou_ = 0.0

    for th, meter in iou_meters.items():
      miou, miou_fg, iou, *_ = meter.get(clear=True, detail=True)
      if miou_ < miou:
        th_ = th
        miou_ = miou
        # miou_fg_ = miou_fg
        iou_ = [round(iou[c], 2) for c in classes]
    
    return th_, miou_, iou_


class Calculator_For_mIoU:
    def __init__(self, classes):
        if isinstance(classes, np.ndarray):
            classes = classes.tolist()

        self.class_names = ['background'] + classes
        self.classes = len(self.class_names)

        self.clear()

    def get_data(self, pred_mask, gt_mask):
        obj_mask = gt_mask<255
        correct_mask = (pred_mask==gt_mask) * obj_mask
        
        P_list, T_list, TP_list = [], [], []
        for i in range(self.classes):
            P_list.append(np.sum((pred_mask==i)*obj_mask))
            T_list.append(np.sum((gt_mask==i)*obj_mask))
            TP_list.append(np.sum((gt_mask==i)*correct_mask))

        return (P_list, T_list, TP_list)

    def add_using_data(self, data):
        P_list, T_list, TP_list = data
        for i in range(self.classes):
            self.P[i] += P_list[i]
            self.T[i] += T_list[i]
            self.TP[i] += TP_list[i]

    def add(self, pred_mask, gt_mask):
        obj_mask = gt_mask<255
        correct_mask = (pred_mask==gt_mask) * obj_mask

        for i in range(self.classes):
            self.P[i] += np.sum((pred_mask==i)*obj_mask)
            self.T[i] += np.sum((gt_mask==i)*obj_mask)
            self.TP[i] += np.sum((gt_mask==i)*correct_mask)

    def get(self, detail=False, clear=True):
        IoU_dic = {}
        IoU_list = []

        FP_list = [] # over activation
        FN_list = [] # under activation

        for i in range(self.classes):
            IoU = self.TP[i]/(self.T[i]+self.P[i]-self.TP[i]+1e-10) * 100
            FP = (self.P[i]-self.TP[i])/(self.T[i] + self.P[i] - self.TP[i] + 1e-10)
            FN = (self.T[i]-self.TP[i])/(self.T[i] + self.P[i] - self.TP[i] + 1e-10)

            IoU_dic[self.class_names[i]] = IoU

            IoU_list.append(IoU)
            FP_list.append(FP)
            FN_list.append(FN)
        
        iou = np.asarray(IoU_list)
        mIoU = np.mean(iou)
        mIoU_foreground = np.mean(iou[1:])

        FP = np.mean(np.asarray(FP_list))
        FN = np.mean(np.asarray(FN_list))
        
        if clear:
            self.clear()
        
        if detail:
            return mIoU, mIoU_foreground, IoU_dic, FP, FN
        else:
            return mIoU, mIoU_foreground

    def clear(self):
        self.TP = []
        self.P = []
        self.T = []
        
        for _ in range(self.classes):
            self.TP.append(0)
            self.P.append(0)
            self.T.append(0)


class MIoUCalcFromNames(Calculator_For_mIoU):
    def __init__(self, class_names):
        self.class_names = class_names
        self.classes = len(self.class_names)
        self.clear()
