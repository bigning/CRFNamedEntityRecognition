import os,sys
import operator

words_count = {}
train_data_path = '../../data/ner/eng.train.new'
train_data_file = open(train_data_path)
train_data_lines = train_data_file.readlines()
train_data_file.close()

words_file = open('../../data/intermedia_data/words.txt', 'w')

train_data_lines = train_data_lines[:-1]

indices= range(0, len(train_data_lines), 3)
for index in indices:
    data = train_data_lines[index]
    data = data.strip('\n')
    data = data.strip(' ')
    data = data.lower()
    data_array = data.split(' ')

    for word in data_array:
        if not words_count.has_key(word):
            words_count[word] = 1
        else:
            words_count[word] += 1

sorted_words_count = sorted(words_count.items(), key = operator.itemgetter(1), reverse = True)
print(len(sorted_words_count))

for item in sorted_words_count:
    words_file.writelines(item[0] + ' ' + str(item[1]) + '\n')
words_file.close()
