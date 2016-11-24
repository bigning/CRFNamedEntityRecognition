#include "feature_extractor.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

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

void FeatureExtractor::extract_feature(int t, int y, int y_prev,
        vector<string>& input, vector<int>& feature) {
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
        extract_feature(i, label[i], label[i - 1], data, feature);
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
    dict_size_ = word_ind;
}

void FeatureExtractor::extract_original_feature(int t, int y, int y_prev, 
        vector<string>& input, vector<int>& feature) {
    int feature_size = 0;
    // label-label feature (see tutorial P300)
    int ll_feature_size = label_num_ * label_num_;
    std::vector<int> ll_feature(ll_feature_size, 0);
    ll_feature[y * label_num_ + y_prev] = 1;
    // end label-label feature
    
    // label-word feature (P300)
    int lw_feature_size = label_num_ * dict_size_;
    std::vector<int> lw_feature(lw_feature_size, 0);
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
    lw_feature[y * dict_size_ + word_ind] = 1;
    // end label-word feature
    
    feature_size = ll_feature_size + lw_feature_size;
    feature = vector<int>(feature_size, 0);
    std::copy(ll_feature.begin(), ll_feature.end(), feature.begin());
    std::copy(lw_feature.begin(), lw_feature.end(),
            feature.begin() + ll_feature_size);
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
        
        std::vector<int> feature;
        for (int i = 1; i < data.size(); i++) {
            extract_original_feature(i, label[i], label[i - 1], data, feature);
            if (feature_sum.size() == 0) {
                feature_sum = feature;
            } else {
                for (int j = 0; j < feature.size(); j++) {
                    feature_sum[j] += feature[j];
                }
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
        << std::endl;
    std::ofstream out(selected_feature_ind_save_filename.c_str());
    for (int i = 0; i < selected_feature_size_; i++) {
        out << selected_feature_ind_[i] << " ";
    }
    out.close();
}

void FeatureExtractor::sentence2input_data(std::string& sentence, vector<string>& data) {
    sentence = "<start> " + sentence;
    std::istringstream iss(sentence);
    std::string tmp_data;
    while (iss >> tmp_data) {
        data.push_back(tmp_data);
    }
}

void FeatureExtractor::tags2lable_data(std::string& tags, vector<int>& label) {
    tags = "O " + tags;
    std::istringstream iss(tags);
    std::string tmp_str;
    while (iss >> tmp_str) {
        label.push_back(label_map_[tmp_str]);
    }
}
