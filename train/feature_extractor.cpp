#include "feature_extractor.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <ctime>
#include <cstring>

using namespace std;

FeatureExtractor::FeatureExtractor() {
    train_data_filename_ = "../../data/ner/eng.train.new";
    lexicon_filename_ = "../../data/intermedia_data/words.txt";
    selected_feature_ind_save_filename = "../../data/intermedia_data/feature_selection.data";

    label_map_["O"] = 0;
    label_map_["B-PER"] = 1;
    label_map_["I-PER"] = 2;
    label_map_["B-LOC"] = 3;
    label_map_["I-LOC"] = 4;
    label_map_["B-ORG"] = 5;
    label_map_["I-ORG"] = 6;
    label_map_["B-MISC"] = 7;
    label_map_["I-MISC"] = 8;

    label_num_ = 9;

    load_words();
    load_selected_feature_index();

    original_feature_ = NULL;
}

void FeatureExtractor::load_selected_feature_index() {
    std::ifstream selected_feature_ind_file(selected_feature_ind_save_filename.c_str());
    std::string line;
    std::getline(selected_feature_ind_file, line);
    selected_feature_ind_file.close();

    std::istringstream iss(line);
    selected_feature_ind_.clear();
    int selected_ind;
    while (iss >> selected_ind) {
        selected_feature_ind_.push_back(selected_ind);
    }
    selected_feature_size_ = selected_feature_ind_.size();
}

void FeatureExtractor::extract_features(vector<std::string>& data, 
        vector<vector<vector<SparseVector> > >& features) {
    features = vector<vector<vector<SparseVector> > >(data.size(), 
            vector<vector<SparseVector> >(label_num_, vector<SparseVector>(
                    label_num_)));

    for (int t = 1; t < data.size() - 1; t++) {
        for (int i = 0; i < label_num_; i++) {
            for (int j = 0; j < label_num_; j++) {
                features[t][i][j] = extract_feature(t, i, j, data);
            }
        }
    }
}

vector<int> FeatureExtractor::extract_feature(int t, int y, int y_prev,
        vector<string>& input) {
    /*
    vector<int> feature;
    extract_original_feature(t, y, y_prev, input, feature);

    for (int i = 0; i < selected_feature_size_; i++) {
        feature[i] = feature[selected_feature_ind_[i]];
    }
    feature.resize(selected_feature_size_);
    int non_zero_num = 0;
    for (int i = 0; i < feature.size(); i++) {
        if (feature[i] != 0) {
            feature[non_zero_num++] = i;
        }
    }
    feature.resize(non_zero_num);
    tmp_feature = feature;
    */

    extract_original_feature(t, y, y_prev, input);
    vector<int> tmp_feature = vector<int>(non_zero_num_, 0);
    int ind = 0;
    for (int i = 0; i < selected_feature_size_; i++) {
        if (original_feature_[selected_feature_ind_[i]] != 0) {
            //feature_2[ind++] = i;
            tmp_feature[ind++] = i;
            //tmp_feature.push_back(i); 
        }
    }
    tmp_feature.resize(ind);
    return tmp_feature;
}

void FeatureExtractor::test_extract_feature() {
    std::ifstream input("../../data/ner/eng.train.new");
    std::string data_str;
    std::string tag_str;
    std::getline(input, data_str);
    std::getline(input, tag_str);
    input.close();

    std::vector<string> data;
    sentence2input_data(data_str, data);
    std::vector<int> label;
    tags2lable_data(tag_str, label);
    vector<int> feature;
    for (int i = 1; i < data.size(); i++) {
        feature = extract_feature(i, label[i], label[i - 1], data);
        std::cout << label[i] << " ";
        for (int j = 0; j < feature.size(); j++) {
            std::cout << feature[j] << " ";
        }
        std::cout << std::endl;
    }
}

void FeatureExtractor::load_words() {
    std::ifstream words_input_file(lexicon_filename_.c_str());
    std::string word;
    int freq = 0;
    int word_ind = 0;
    while (words_input_file >> word >> freq) {
        if (freq <= 1) {
            continue;
        }
        word2ind_.insert(std::pair<std::string, int>(word, word_ind));
        word_ind++;
    }
    words_input_file.close();
    word2ind_["<start>"] = word_ind++;
    word2ind_["<rare_word>"] = word_ind++;
    word2ind_["<stop>"] = word_ind++;
    dict_size_ = word_ind;
}

void FeatureExtractor::extract_original_feature(int t, int y, int y_prev, 
        vector<string>& input) {

    // adding new feature pipeline:
    // 1. feature size
    // 2. cal feature
    // 3. update non-zero num
    // 4. update feature_ind
    
    non_zero_num_ = 0;
    
    int ll_feature_size = label_num_ * label_num_;
    //lw_1
    int lw_feature_size = label_num_ * dict_size_;
    int lw_0_feature_size = lw_feature_size;
    int lw_2_feature_size = lw_feature_size;
    // Named Entity Recognition with a Maximum Entropy Approach
    int first_word_init_caps_feature_size = label_num_;
    int init_caps_feature_size = label_num_;
    int first_word_init_not_caps_feature_size = label_num_;
    int is_all_caps_feature_size = label_num_;
    int prev_init_caps_feature_size = label_num_;
    int next_init_caps_feature_size = label_num_;
    int case_sequence_feature_size = label_num_;

    int feature_ind = 0;

    int feature_size = ll_feature_size + lw_feature_size + lw_0_feature_size +
        lw_2_feature_size + first_word_init_caps_feature_size + init_caps_feature_size +
        first_word_init_not_caps_feature_size + is_all_caps_feature_size +
        prev_init_caps_feature_size + next_init_caps_feature_size + 
        case_sequence_feature_size;
    original_feature_size_ = feature_size;
    //feature = vector<int>(feature_size, 0);
    if (original_feature_ == NULL) {
        original_feature_ = new int[feature_size];
    }
    std::memset(original_feature_, 0, feature_size*sizeof(int));
    int* feature = original_feature_;

    // label-label feature (see tutorial P300)
    feature[feature_ind + y * label_num_ + y_prev] = 1;
    non_zero_num_ += 1;
    feature_ind += ll_feature_size;
    // end label-label feature
    
    // label-word feature (P300) lw_1
    std::string word = input[t];
    for (int i = 0; i < word.size(); i++) {
        if (word[i] >= 'A' && word[i] <= 'Z') {
            word[i] += 32;
        }
    }
    int word_ind = 0;
    std::map<std::string, int>::iterator iter = word2ind_.find(word);
    if (iter == word2ind_.end()) {
        word_ind = word2ind_["<rare_word>"];
    } else {
        word_ind = iter->second;
    }
    feature[feature_ind + y * dict_size_ + word_ind] = 1;
    non_zero_num_ += 1;
    feature_ind += lw_feature_size;
    // end label-word feature
    
    // label-word feature (P300) lw_0
    if (t - 1 >= 0) {
        std::string word = input[t - 1];
        for (int i = 0; i < word.size(); i++) {
            if (word[i] >= 'A' && word[i] <= 'Z') {
                word[i] += 32;
            }
        }
        word_ind = 0;
        std::map<std::string, int>::iterator iter = word2ind_.find(word);
        if (iter == word2ind_.end()) {
            word_ind = word2ind_["<rare_word>"];
        } else {
            word_ind = iter->second;
        }
        feature[feature_ind + y * dict_size_ + word_ind] = 1;
        non_zero_num_ += 1;
    }
    feature_ind += lw_0_feature_size;
    // end label-word feature
    
    // label-word feature (P300) lw_2
    if (t + 1 <= input.size() - 1) {
        std::string word = input[t + 1];
        for (int i = 0; i < word.size(); i++) {
            if (word[i] >= 'A' && word[i] <= 'Z') {
                word[i] += 32;
            }
        }
        word_ind = 0;
        std::map<std::string, int>::iterator iter = word2ind_.find(word);
        if (iter == word2ind_.end()) {
            word_ind = word2ind_["<rare_word>"];
        } else {
            word_ind = iter->second;
        }
        feature[feature_ind + y * dict_size_ + word_ind] = 1;
        non_zero_num_ += 1;
    }
    feature_ind += lw_2_feature_size;
    // end label-word feature
    
    //first_word_init_caps_feature
    if (t == 1 && (input[t][0] >= 'A' && input[t][0] <= 'Z')) {
        feature[feature_ind + y] = 1;
        non_zero_num_ += 1;
    }
    feature_ind += first_word_init_caps_feature_size; 
    // end first_word_init_caps_feature
    
    // init_caps_feature
    bool init_caps = false;
    if (input[t][0] >= 'A' && input[t][0] <= 'Z' && t != 1) {
        feature[feature_ind + y] = 1;
        non_zero_num_ += 1;
        init_caps = true;
    }
    feature_ind += init_caps_feature_size;
    // end init_caps_feature
    
    //first_word_not_init_caps_feature
    if (t == 1 && (input[t][0] < 'A' || input[t][0] > 'Z')) {
        feature[feature_ind + y] = 1;
        non_zero_num_ += 1;
    }
    feature_ind += first_word_init_not_caps_feature_size; 
    // end first_word_not_init_caps_feature
    
    // is_all_caps feature
    bool is_all_caps = true;
    for (int i = 0; i < input[t].size(); i++) {
        if (input[t][i] < 'A' || input[t][i] > 'Z') {
            is_all_caps = false;
        }
    }
    if (is_all_caps) {
        feature[feature_ind + y] = 1;
        non_zero_num_ += 1;
    }
    feature_ind += is_all_caps_feature_size;
    // end is_all_caps
    
    // prev_init_caps_feature
    bool prev_init_caps = false;
    if (t - 1 >= 0) {
        if (input[t-1][0] >= 'A' && input[t - 1][0] <= 'Z') {
            feature[feature_ind + y] = 1;
            prev_init_caps = true;
            non_zero_num_ += 1;
        }
    }
    feature_ind += prev_init_caps_feature_size;
    // end prev_init_caps_feature
    
    // next_init_caps_feature
    bool next_init_caps = false;
    if (t + 1 < input[t].size()) {
        if (input[t + 1][0] >= 'A' && input[t + 1][0] <= 'Z') {
            feature[feature_ind + y] = 1;
            next_init_caps = true;
            non_zero_num_ += 1;
        }
    }
    feature_ind += next_init_caps_feature_size;
    // end next_init_caps_feature
    
    // case sequence feature
    if (prev_init_caps && next_init_caps && init_caps) {
        feature[feature_ind + y] = 1;
        non_zero_num_ += 1;
    }
    feature_ind += case_sequence_feature_size;
    // end case sequence feature
    
    /*
    end = clock();
    std::cout << "time_2: " << (double)(end - begin) / CLOCKS_PER_SEC << std::endl;;
    
    for (int i = 0; i < feature.size(); i++) {
        if (feature[i] != 0) {
            std::cout << i << " ";
        }
    }
    std::cout << std::endl;
    */
}

void FeatureExtractor::get_selected_feature_index() {
    std::ifstream train_data_file(train_data_filename_.c_str());
    std::string data_line, label_line, empty_line;
    std::vector<int> feature_sum;
    int ind = 0;
    while (std::getline(train_data_file, data_line)) {
        std::getline(train_data_file, label_line);
        std::getline(train_data_file, empty_line);

        std::vector<std::string> data;
        std::vector<int> label;
        sentence2input_data(data_line, data);
        tags2lable_data(label_line, label);
        
        for (int i = 1; i < data.size() - 1; i++) {
            extract_original_feature(i, label[i], label[i - 1], data);
            if (feature_sum.size() == 0) {
                feature_sum = vector<int>(original_feature_size_, 0);
            } 
            for (int j = 0; j < original_feature_size_; j++) {
                feature_sum[j] += original_feature_[j];
            }
        }
        ind++;
        if (ind % 100 == 0) {
            std::cout << ind << std::endl;
        }
    }
    train_data_file.close();
    selected_feature_ind_ = std::vector<int>(feature_sum.size(), -1);
    selected_feature_size_ = 0; 
    for (int i = 0; i < feature_sum.size(); i++) {
        if (feature_sum[i] >= 2) {
            selected_feature_ind_[selected_feature_size_++] = i;
        }
    }
    selected_feature_ind_.resize(selected_feature_size_);
    std::cout << "[INFO]: " << selected_feature_size_ << " features are selected" 
        << "from " << feature_sum.size() << std::endl;
    std::ofstream out(selected_feature_ind_save_filename.c_str());
    for (int i = 0; i < selected_feature_size_; i++) {
        out << selected_feature_ind_[i] << " ";
    }
    out.close();
}

void FeatureExtractor::sentence2input_data(std::string& sentence, vector<string>& data) {
    data.clear();
    sentence = "<start> " + sentence + " <stop>";
    std::istringstream iss(sentence);
    std::string tmp_data;
    while (iss >> tmp_data) {
        data.push_back(tmp_data);
    }
}

void FeatureExtractor::tags2lable_data(std::string& tags, vector<int>& label) {
    label.clear();
    tags = "O " + tags + " O";
    std::istringstream iss(tags);
    std::string tmp_str;
    while (iss >> tmp_str) {
        label.push_back(label_map_[tmp_str]);
    }
}
