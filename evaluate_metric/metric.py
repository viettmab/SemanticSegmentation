from collections import OrderedDict

import torch as th

def f_score(precision, recall, beta=1):
    """calculate the f-score value.
    Args:
        precision (float | torch.Tensor): The precision value.
        recall (float | torch.Tensor): The recall value.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.
    Returns:
        [torch.tensor]: The f-score value.
    """
    score = (1 + beta**2) * (precision * recall) / (
        (beta**2 * precision) + recall)
    return score

def intersect_and_union(pred_label,
                        label,
                        num_classes=19,
                        ignore_index=255,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and Union.
    Args:
        pred_label (tensor N*H*W): Prediction segmentation map
        label (tensor N*H*W): Ground truth segmentation map
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. The parameter will
            work only when label is str. Default: False.
     Returns:
         th.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         th.Tensor: The union of prediction and ground truth histogram on
            all classes.
         th.Tensor: The prediction histogram on all classes.
         th.Tensor: The ground truth histogram on all classes.
    """

    if label_map is not None:
        label_copy = label.clone()
        for old_id, new_id in label_map.items():
            label[label_copy == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect = th.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = th.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = th.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label

def total_area_to_metrics(total_area_intersect,
                          total_area_union,
                          total_area_pred_label,
                          total_area_label,
                          metrics=['mIoU'],
                          nan_to_num=None,
                          beta=1):
    """Calculate evaluation metrics
    Args:
        total_area_intersect (tensor): The intersection of prediction and
            ground truth histogram on all classes.
        total_area_union (tensor): The union of prediction and ground truth
            histogram on all classes.
        total_area_pred_label (tensor): The prediction histogram on all
            classes.
        total_area_label (tensor): The ground truth histogram on all classes.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        tensor: Per category accuracy, shape (num_classes, ).
        tensor: Per category evaluation metrics, shape (num_classes, ).
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice', 'mFscore']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))

    all_acc = total_area_intersect.sum() / total_area_label.sum()
    ret_metrics = OrderedDict({'aAcc': all_acc})
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            acc = total_area_intersect / total_area_label
            ret_metrics['IoU'] = iou
            ret_metrics['Acc'] = acc
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            acc = total_area_intersect / total_area_label
            ret_metrics['Dice'] = dice
            ret_metrics['Acc'] = acc
        elif metric == 'mFscore':
            precision = total_area_intersect / total_area_pred_label
            recall = total_area_intersect / total_area_label
            f_value = th.tensor(
                [f_score(x[0], x[1], beta) for x in zip(precision, recall)])
            ret_metrics['Fscore'] = f_value
            ret_metrics['Precision'] = precision
            ret_metrics['Recall'] = recall

#     ret_metrics = {
#         metric: value.numpy()
#         for metric, value in ret_metrics.items()
#     }
#     if nan_to_num is not None:
#         ret_metrics = OrderedDict({
#             metric: np.nan_to_num(metric_value, nan=nan_to_num)
#             for metric, metric_value in ret_metrics.items()
#         })
    return ret_metrics