from sklearn.metrics import confusion_matrix


def precision(gt, mask):
    gt = gt.flatten()
    mask = mask.flatten()
    tn, fp, fn, tp = confusion_matrix(gt, mask).ravel()
    prec = tp / (tp + fp)
    return (prec)


####recall---
def recall(gt, mask):
    gt = gt.flatten()
    mask = mask.flatten()
    tn, fp, fn, tp = confusion_matrix(gt, mask).ravel()
    rec = tp / (tp + fn)
    return (rec)


###f1 score--

def f1_score(prec, rec):
    f1 = 2 * (prec * rec) / (prec + rec)
    return f1

    ### jaccard


def jaccard(gt, mask):
    gt = gt.flatten()
    mask = mask.flatten()
    tn, fp, fn, tp = confusion_matrix(gt, mask).ravel()
    rec = tp / (tp + fn + fp)
    return (rec)

    ### jaccard


def Overall(gt, mask):
    gt = gt.flatten()
    mask = mask.flatten()
    tn, fp, fn, tp = confusion_matrix(gt, mask).ravel()
    rec = (tp + tn) / (tp + fp + fn + tn)
    return (rec)


###aji score

def get_fast_aji(true, pred):
    true = np.copy(true)  # ? do we need this
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [None, ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [None, ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_inter = np.zeros([len(true_id_list) - 1,
                               len(pred_id_list) - 1], dtype=np.float64)
    pairwise_union = np.zeros([len(true_id_list) - 1,
                               len(pred_id_list) - 1], dtype=np.float64)

    # caching pairwise
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[int(true_id)]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[int(pred_id)]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[int(true_id) - 1, int(pred_id) - 1] = inter
            pairwise_union[int(true_id) - 1, int(pred_id) - 1] = total - inter
    #
    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    # pair of pred that give highest iou for each true, dont care
    # about reusing pred instance multiple times
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    # print(paired_true.shape, paired_pred.shape)
    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()
    #
    paired_true = (list(paired_true + 1))  # index to instance ID
    paired_pred = (list(paired_pred + 1))
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array([idx for idx in true_id_list[1:] if idx not in paired_true])
    unpaired_pred = np.array([idx for idx in pred_id_list[1:] if idx not in paired_pred])
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()
    #
    aji_score = overall_inter / overall_union
    return aji_score


sum = 0
for i in range(len(Y_test)):
    sum = sum + precision(Y_test[i], preds_test_t[i])
prec = sum / len(Y_test)

sum = 0
for i in range(len(Y_test)):
    sum = sum + recall(Y_test[i], preds_test_t[i])
rec = sum / len(Y_test)

sum = 0
for i in range(len(Y_test)):
    sum = sum + jaccard(Y_test[i], preds_test_t[i])
jaccard1 = sum / len(Y_test)

sum = 0
for i in range(len(Y_test)):
    sum = sum + Overall(Y_test[i], preds_test_t[i])
Overall1 = sum / len(Y_test)

f1 = f1_score(prec, rec)
aji = get_fast_aji(Y_test, preds_test_t)

print("Jaccard Index", jaccard1)
print("final f1", f1)
print("final precision", prec)
print("final recall", rec)
print("Overall Accuracy", Overall1)
print("final aji", aji)