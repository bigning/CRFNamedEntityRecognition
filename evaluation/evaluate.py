import os,sys

def get_original_label_from_gt(gt_arr):
    label = {}
    prev_status = ''
    prev_label = ''
    prev_ind = []
    for ind, ele in enumerate(gt_arr):
        if ele == "O":
            if prev_label != '':
                if prev_label in label:
                    label[prev_label].append(tuple(prev_ind))
                else:
                    label[prev_label] = [tuple(prev_ind)]
            prev_label = ''
            prev_status = ''
            prev_ind = []
            continue
        if ele.startswith('B-'):
            if prev_label != '':
                if prev_label in label:
                    label[prev_label].append(tuple(prev_ind))
                else:
                    label[prev_label] = [tuple(prev_ind)]
            prev_label = ele[2:]
            prev_status = 'B-'
            prev_ind = [ind]
            continue
        if ele.startswith('I-'):
            cur_label = ele[2:]
            if cur_label != prev_label or prev_label == '':
                if prev_label in label:
                    label[prev_label].append(tuple(prev_ind))
                else:
                    label[prev_label] = [tuple(prev_ind)]
                prev_label = ele[2:]
                prev_status = 'I-'
                prev_ind = [ind]
                continue
            prev_status = 'I-'
            prev_ind.append(ind)
    if prev_label != "":
        if prev_label in label:
            label[prev_label].append(tuple(prev_ind))
        else:
            label[prev_label] = [tuple(prev_ind)]

    return label

gt_filename = "../../data/ner/eng.testb.new"
prediction_filename = gt_filename + ".res"

gt_file = open(gt_filename)
gt_lines = gt_file.readlines()
gt_file.close()

prediction_file = open(prediction_filename)
prediction_lines = prediction_file.readlines()
prediction_file.close()

original_labels = ["LOC", "PER", "ORG", "MISC", "O"]
bio_labels = ["B-LOC", "I-LOC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-MISC", "I-MISC", "O"]
bio_gt_num = {}
bio_gt_recall = {}
bio_pred_num = {}
bio_pred_precision = {}

for label in bio_labels:
    bio_gt_num[label] = 0
    bio_gt_recall[label] = 0
    bio_pred_num[label] = 0
    bio_pred_precision[label] = 0

original_gt_num = {}
original_gt_recall = {}
original_pred_num = {}
original_pred_precision = {}
for label in original_labels:
    original_gt_num[label] = 0
    original_gt_recall[label] = 0
    original_pred_num[label] = 0
    original_pred_precision[label] = 0

original_gt2pred = {}
original_pred2gt = {}
for label in original_labels:
    original_gt2pred[label] = {}
    original_pred2gt[label] = {}
    for label_2 in original_labels:
        original_gt2pred[label][label_2] = 0
        original_pred2gt[label][label_2] = 0

bio_gt2pred = {}
bio_pred2gt = {}
for label in bio_labels:
    bio_gt2pred[label] = {}
    bio_pred2gt[label] = {}
    for label_2 in bio_labels:
        bio_gt2pred[label][label_2] = 0
        bio_pred2gt[label][label_2] = 0

data_ind = range(0, len(gt_lines), 3)
for ind in data_ind:
    sentence = gt_lines[ind].strip('\n').strip(' ')
    gt_str = gt_lines[ind + 1].strip('\n').strip(' ')
    pred_str = prediction_lines[ind + 1].strip('\n').strip(' ')

    gt_str_arr = gt_str.split(' ')
    pred_str_arr = pred_str.split(' ')

    for gt, pred in zip(gt_str_arr, pred_str_arr):
        bio_gt2pred[gt][pred] += 1
        bio_pred2gt[pred][gt] += 1
    
    original_gt = get_original_label_from_gt(gt_str_arr)
    original_pred = get_original_label_from_gt(pred_str_arr)
    for key in original_gt:
        for ele in original_gt[key]:
            if original_pred.has_key(key) and ele in original_pred[key]:
                original_gt2pred[key][key] += 1
            else:
                original_gt2pred[key]["O"] += 1
    for key in original_pred:
        for ele in original_pred[key]:
            if original_gt.has_key(key) and ele in original_gt[key]:
                original_pred2gt[key][key] += 1
            else:
                original_pred2gt[key]["O"] += 1

bio_recall_sum = 0.0
bio_recall = 0.0
bio_precision_sum = 0.0
bio_precision = 0.0
print("BIO evaluation: ")
print("gt2pred: ")
tmp_str = ""
for key in bio_gt2pred:
    tmp_str += key + "\t"
print tmp_str
for key_1 in bio_gt2pred:
    tmp_str = ''
    bio_gt_recall[key_1] += bio_gt2pred[key_1][key_1]
    if key_1 != "O":
        bio_recall += bio_gt2pred[key_1][key_1]
    for key_2 in bio_gt2pred[key_1]:
        tmp_str += str(bio_gt2pred[key_1][key_2]) + "\t"
        bio_gt_num[key_1] += bio_gt2pred[key_1][key_2]
        if key_1 != "O":
            bio_recall_sum += bio_gt2pred[key_1][key_2]
    print tmp_str
print(' ')
print("pred2gt: ")
for key_1 in bio_pred2gt:
    tmp_str = ''
    bio_pred_precision[key_1] += bio_pred2gt[key_1][key_1]
    if key_1 != "O":
        bio_precision += bio_pred2gt[key_1][key_1]
    for key_2 in bio_pred2gt[key_1]:
        tmp_str += str(bio_pred2gt[key_1][key_2]) + "\t"
        bio_pred_num[key_1] += bio_pred2gt[key_1][key_2]
        if key_1 != "O":
            bio_precision_sum += bio_pred2gt[key_1][key_2]
    print tmp_str

print ("bio precison: " + str(bio_precision) + "/" + str(bio_precision_sum) + "= " + str(bio_precision / bio_precision_sum))
print ("bio recall: " + str(bio_recall) + "/" + str(bio_recall_sum) + "= " + str(bio_recall/ bio_recall_sum))


original_recall_sum = 0.0
original_recall = 0.0
original_precision_sum = 0.0
original_precision = 0.0
print("ORIGINAL evaluation: ")
print("gt2pred: ")
tmp_str = ""
for key in original_gt2pred:
    tmp_str += key + "\t"
print tmp_str
for key_1 in original_gt2pred:
    tmp_str = ''
    original_gt_recall[key_1] += original_gt2pred[key_1][key_1]
    if key_1 != "O":
        original_recall += original_gt2pred[key_1][key_1]
    for key_2 in original_gt2pred[key_1]:
        tmp_str += str(original_gt2pred[key_1][key_2]) + "\t"
        original_gt_num[key_1] += original_gt2pred[key_1][key_2]
        if key_1 != "O":
            original_recall_sum += original_gt2pred[key_1][key_2]
    print tmp_str
print(' ')
print("pred2gt: ")
tmp_str = ""
for key in original_pred2gt:
    tmp_str += key + "\t"
print tmp_str
for key_1 in original_pred2gt:
    tmp_str = ''
    original_pred_precision[key_1] += original_pred2gt[key_1][key_1]
    if key_1 != "O":
        original_precision += original_pred2gt[key_1][key_1]
    for key_2 in original_pred2gt[key_1]:
        tmp_str += str(original_pred2gt[key_1][key_2]) + "\t"
        original_pred_num[key_1] += original_pred2gt[key_1][key_2]
        if key_1 != "O":
            original_precision_sum += original_pred2gt[key_1][key_2]
    print tmp_str

precision = original_precision /  original_precision_sum
print ("original precison: " + str(original_precision) + "/" + str(original_precision_sum) + "= " + str(precision))
recall = original_recall / original_recall_sum
print ("original recall: " + str(original_recall) + "/" + str(original_recall_sum) + "= " + str(recall))
f_score = 2 * (precision * recall) / (precision + recall)
print("f1 score: " + str(f_score))
