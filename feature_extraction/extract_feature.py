import os,sys
import operator
import numpy
import timeit

class FeatureExtractor:

    def __init__(self):
        self.lexicon = {}
        self.lexicon_size = 0
        self.label_map = {'O': 0, 'B-PER':1, 'I-PER':2, 'B-LOC':3, 'I-LOC':4, \
                'B-ORG':5, 'I-ORG':6, 'B-MISC':7, 'I-MISC':8}
        self.label_num = 9
        self.selected_feature_ind = []

        self.load_words()
        self.load_selected_feature_index()

    def load_selected_feature_index(self):
        selected_feature_ind_file = open('../../data/intermedia_data/feature_selection.data')
        lines = selected_feature_ind_file.readlines()
        selected_feature_ind_file.close()
        line = lines[0].strip(' \n')
        line_arr = line.split(' ')
        line_arr = [int(num) for num in line_arr]
        self.selected_feature_ind = numpy.array(line_arr, dtype=int)

    def load_words(self):
        words_file = open('../../data/intermedia_data/words.txt', 'r')
        words_lines = words_file.readlines()
        words_file.close()

        index = 0
        for line in words_lines:
            line = line.strip('\n')
            line_arr = line.split(' ')
            freq = int(line_arr[1])
            ## remove words occuring only once
            if freq <= 1:
                continue
            self.lexicon[line_arr[0]] = index
            
            index += 1

        self.lexicon['<start>'] = index
        index += 1
        self.lexicon['new_word'] = index
        index += 1
        self.lexicon_size = index

    def batch_extract_original_training_features(self):
        training_data_file = open('../../data/ner/eng.train.new')
        lines = training_data_file.readlines()
        training_data_file.close()
        lines = lines[:-1]

        indices = range(0, len(lines), 3)
        feature_sum = []
        sentence_num = 0
        max_ind = 0
        for index in indices:
            inputs = lines[index].strip('\n')
            labels = lines[index + 1].strip('\n')
            inputs = '<start> ' + inputs
            inputs = inputs.split(' ')
            labels = 'O ' + labels
            labels_arr = labels.split(' ')
            labels = []
            for label in labels_arr:
                labels.append(self.label_map[label])
            
            for t in range(1, len(inputs)):
                feature = self.extract_original_feature(t,labels[t], labels[t-1],inputs)
                sparse_feature = numpy.nonzero(feature)[0]
                max_ind = max(max_ind, numpy.amax(sparse_feature))
                if feature_sum == []:
                    feature_sum = feature
                else:
                    feature_sum += feature
            sentence_num += 1
            if (sentence_num % 100 == 0):
                print sentence_num
        
        
        zero_num = 0
        one_num = 0
        for index in range(len(feature_sum)):
            if feature_sum[index] == 0:
                zero_num += 1
            elif feature_sum[index] == 1:
                one_num += 1
        print('one_num: ' + str(one_num))
        print('zero_num: ' + str(zero_num))
        print ('feature dimension: ' + str(feature_sum.size))
        print max_ind

        ## select feature which occurs at least twice
        ## ref: Named Entity Recognition with a Maximum Entropy Approach
        non_zero_feature_ind = numpy.where(feature_sum > 1)
        non_zero_feature_ind = non_zero_feature_ind[0]
        non_zero_feature_file = open('../../data/intermedia_data/feature_selection.data', 'w')
        feature_selected_str = '' 
        for i in range(0, len(non_zero_feature_ind)):
            feature_selected_str += str(non_zero_feature_ind[i]) + ' '
        non_zero_feature_file.writelines(feature_selected_str)

    def batch_extract_training_features(self):
        training_data_file = open('../../data/ner/eng.train.new')
        lines = training_data_file.readlines()
        training_data_file.close()
        lines = lines[:-1]

        sparse_feature_file = open('../../data/intermedia_data/feature.train', 'w')

        indices = range(0, len(lines), 3)
        feature_sum = []
        sentence_num = 0
        for index in indices:
            inputs = lines[index].strip('\n')
            labels = lines[index + 1].strip('\n')
            inputs = '<start> ' + inputs
            inputs = inputs.split(' ')
            labels = 'O ' + labels
            labels_arr = labels.split(' ')
            labels = []
            for label in labels_arr:
                labels.append(self.label_map[label])
            
            write_str = ''
            for t in range(1, len(inputs)):
                feature = self.extract_feature(t,labels[t], labels[t-1],inputs)
                sparse_feature = numpy.nonzero(feature)[0]
                if len(sparse_feature) == 0:
                    print '00000'
                    print t
                    print inputs
                    sys.exit()
                write_str += str(labels[t]) + ' '
                for x in numpy.nditer(sparse_feature):
                    write_str += str(x) + ' '
                write_str += '\n'
                if feature_sum == []:
                    feature_sum = feature
                else:
                    feature_sum += feature
            sparse_feature_file.writelines(write_str + '\n')
            sentence_num += 1
            if (sentence_num % 100 == 0):
                print sentence_num
        
        
        zero_num = 0
        one_num = 0
        for index in range(len(feature_sum)):
            if feature_sum[index] == 0:
                zero_num += 1
            elif feature_sum[index] == 1:
                one_num += 1
        print('one_num: ' + str(one_num))
        print('zero_num: ' + str(zero_num))
        sparse_feature_file.close()

    """
    f(y_t, y_t-1, x)
    t: t-th word
    y: t-th label
    y_prev: (t-1)th label
    x: input
    this is before feature selection(select feature that occurs at least twice)
    """
    def extract_original_feature(self, t, y, y_prev, x):
        feature = []

        ## label-label feature (see tutorial P300)
        ll_feature = numpy.zeros(self.label_num * self.label_num)
        ll_feature[y * self.label_num + y_prev] = 1
        ## end label-label feature

        ## label-word feature (see tutorial P300)
        lw_feature = numpy.zeros(self.label_num * self.lexicon_size)
        word_ind = 0
        is_rare = 0
        if x[t].lower() in self.lexicon:
            word_ind = self.lexicon[x[t].lower()]
        else:
            word_ind = self.lexicon_size - 1
            is_rare = 1
        lw_feature[y * self.lexicon_size + word_ind] = 1
        ## end label-word feature

        feature = numpy.concatenate((ll_feature, lw_feature))
        return feature

    def extract_feature(self, t, y, y_prev, x):
        feature = self.extract_original_feature(t, y, y_prev, x)
        feature = feature[self.selected_feature_ind]
        return feature


if __name__ == '__main__':
    feature_extractor = FeatureExtractor()
    if len(sys.argv) == 2:
        print 'extract orginial features'
        feature_extractor.batch_extract_original_training_features()
        sys.exit()
    feature_extractor.batch_extract_training_features()
    print 'hello world'
