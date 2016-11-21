import sys,os

train_data_path = '../../data/ner/eng.train'
train_f = open(train_data_path, 'r')
train_lines = train_f.readlines()
train_f.close()

new_train_data_path = '../../data/ner/eng.train.new'
new_train_f = open(new_train_data_path, 'w')

previous_label = ''
data_str = ''
label_str = ''
index = 0
for line in train_lines:
    line = line.strip('\n')
    line_arr = line.split(' ')
    if len(line_arr) < 2 or line_arr[0] == '-DOCSTART-':
        if (len(data_str) > 0):
            data_str = data_str[:-1] + '\n'
            label_str = label_str[:-1] + '\n'
            new_train_f.writelines(data_str + label_str + '\n')
        previous_label = ''
        data_str = ''
        label_str = ''
        continue

    word = line_arr[0]
    original_label = line_arr[3]
    new_label = original_label

    if (original_label == 'I-ORG' or original_label == 'I-LOC' or \
            original_label == 'I-PER' or original_label == 'I-MISC') and \
            original_label != previous_label:
        new_label = original_label
        original_label_arr = original_label.split('-')
        new_label = 'B-' + original_label_arr[1]

    if (original_label == 'I-ORG' or original_label == 'I-LOC' or \
            original_label == 'I-PER' or original_label == 'I-MISC') and \
            original_label == previous_label:
        new_label = original_label

    data_str += word + ' '
    label_str += new_label + ' '
    previous_label = original_label
    index += 1
    if index % 1000 == 0:
        print index
