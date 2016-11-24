#include "train.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdlib>

Trainer::Trainer() {
    train_data_file_ = "../../data/intermedia_data/feature.train";
    train_num_ = 0;
    feature_length_ = 0;
}

void Trainer::load_train_data() {
    int max_non_zero_index = 0;

    std::ifstream train_data_file(train_data_file_.c_str());
    std::string line;
    int tmp_label = 0;
    int tmp_ind = 0;
    SparseVector tmp_data;
    std::vector<SparseVector> sentence_data;
    std::vector<int> sentence_label;
    while (std::getline(train_data_file, line)) {
        if (line.size() == 0) {
            sentence_data.clear();
            sentence_label.clear();

            train_data_.push_back(DataLabel(sentence_data, sentence_label));
            train_num_++;

            if (sentence_data.size() != sentence_label.size()) {
                std::cout << "[ERROR]: label.size() != data.size()" << std::endl;
                exit(-1);
            }
            continue;
        }
        std::istringstream iss(line);
        iss >> tmp_label;
        tmp_data.clear();
        while (iss >> tmp_ind) {
            tmp_data.push_back(tmp_ind);
            //std::cout << tmp_ind << " ";
            if (tmp_ind > max_non_zero_index) {
                max_non_zero_index = tmp_ind;
            }
        }

        sentence_label.push_back(tmp_label);
        sentence_data.push_back(tmp_data);
    }

    train_data_file.close();
    feature_length_ = max_non_zero_index + 1;
}

void Trainer::weight_initialization() {
    weights_ = std::vector<double>(feature_length_, 0.0);
    
    for (int i = 0; i < weights_.size(); i++) {
        weights_[i] = ((double)rand()) / RAND_MAX;
    }

    gradients_ = std::vector<double>(feature_length_, 0.0);
}

void Trainer::cal_log_alpha(int i) {
    DataLabel& data_label = train_data_[i];
    int T = data_label.label.size();
    double log_alpha_1 = 0;
}

void Trainer::run() {
    load_train_data();
    std::cout << "[INFO]: load " << train_num_ << " train samples. Feature dimension: "
        << feature_length_ << std::endl << std::endl;;

    weight_initialization();
    std::cout << "[INFO]: weights are randomly initialized!" << std::endl << std::endl;
}

int main() {
    //
    //
    srand(0);
    Trainer* p_trainer = new Trainer();
    p_trainer->run();
    std::cout << "hello world" << std::endl;
    return 0;
}
