#include "predictor.h"

#include <fstream>
#include <iostream>
#include <cstdlib>

using std::vector;

Predictor::Predictor(std::string& model) {
    //model_filename_ = "../../data/intermedia_data/model.data";
    model_filename_ = model;

    label_to_tag_[0] = "O";
    label_to_tag_[1] = "B-PER";
    label_to_tag_[2] = "I-PER";
    label_to_tag_[3] = "B-LOC";
    label_to_tag_[4] = "I-LOC";
    label_to_tag_[5] = "B-ORG";
    label_to_tag_[6] = "I-ORG";
    label_to_tag_[7] = "B-MISC";
    label_to_tag_[8] = "I-MISC";

    load_model();
}

void Predictor::load_model() {
    p_feature_extractor_ = new FeatureExtractor();

    std::ifstream model_file(model_filename_.c_str());
    weights_ = std::vector<double>(p_feature_extractor_->selected_feature_size());
    int iter;
    model_file >> iter;
    double tmp_weight;
    int weight_length = 0;
    while (model_file >> tmp_weight) {
        weights_[weight_length++] = tmp_weight;
    }
    model_file.close();
    if (weight_length != p_feature_extractor_->selected_feature_size()) {
        std::cout << "[ERROR]: feature_extractor::selected_feature_size != weight_size"
            << std::endl;
        exit(-1);
    }
}

std::vector<std::string> Predictor::predict(std::string input) {
    std::vector<std::string> data;
    std::vector<int> label; 
    vector<vector<vector<SparseVector> > > features;
    p_feature_extractor_->sentence2input_data(input, data);
    p_feature_extractor_->extract_features(data, features);

    data_label_ = DataLabel(data, label, features);

    vector<vector<double> > log_lambda;
    cal_log_lambda(log_lambda);
    return decode(log_lambda);
}

void Predictor::cal_log_lambda(vector<vector<double> >& log_lambda) {
    int T = data_label_.data.size() - 2;
    log_lambda = std::vector<std::vector<double> >(data_label_.data.size(),
            std::vector<double>(p_feature_extractor_->label_num(), -1));

    // log_lambda_1_j
    for (int j = 0; j < p_feature_extractor_->label_num(); j++) {
        SparseVector& feature = data_label_.features[1][j][0];
        log_lambda[1][j] = weight_time_feature(weights_, feature);
    }

    for (int t = 2; t <= T; t++) {
        for (int j = 0; j < p_feature_extractor_->label_num(); j++) {
            double tmp_max = 0;
            // i = 0
            // log(fai(j, i=0, xt)) + log(alpha_t-1_i). ref: notes P2
            SparseVector& tmp_feature = data_label_.features[t][j][0];
            tmp_max = weight_time_feature(weights_, tmp_feature) + log_lambda[t - 1][0];
            for (int i = 1; i < p_feature_extractor_->label_num(); i++) {
                double tmp_2 = 0;
                SparseVector& tmp_feature = data_label_.features[t][j][i];
                tmp_2 = weight_time_feature(weights_, tmp_feature) + log_lambda[t - 1][i];
                if (tmp_2 > tmp_max) {
                    tmp_max = tmp_2;
                }
            }
            log_lambda[t][j] = tmp_max;
        }
    }
}

std::vector<std::string> Predictor::decode(vector<vector<double> >& log_lambda) {
    // y_T
    int T = data_label_.data.size() - 2;
    std::vector<int> res(T + 2, -1);
    double max_lambda = log_lambda[T][0];
    int max_ind = 0;
    for (int i = 1; i < p_feature_extractor_->label_num(); i++) {
        if (log_lambda[T][i] > max_lambda) {
            max_lambda = log_lambda[T][i];
            max_ind = i;
        }
    }
    res[T] = max_ind;

    for (int t = T - 1; t > 0; t--) {
        int max_ind = 0;
        SparseVector& feature = data_label_.features[t][res[t + 1]][0];
        double tmp_max = weight_time_feature(weights_, feature) + log_lambda[t][0];
        for (int i = 1; i < p_feature_extractor_->label_num(); i++) {
            SparseVector& feature = data_label_.features[t][res[t + 1]][i];
            double tmp_2 = weight_time_feature(weights_, feature) + log_lambda[t][i];
            if (tmp_2 > tmp_max) {
                tmp_max = tmp_2;
                max_ind = i;
            }
        }
        res[t] = max_ind;
    }
    std::vector<std::string> pred_lable;
    for (int t = 1; t <= T; t++) {
        pred_lable.push_back(label_to_tag_[res[t]]);
    }
    return pred_lable;
}

void Predictor::batch_predict(const std::string& input_filename) {
    std::ifstream input(input_filename.c_str());
    std::ofstream output((input_filename + ".res").c_str());

    std::string sentence;
    std::string tmp;
    int ind = 0;
    while (std::getline(input, sentence)) {
        std::getline(input, tmp);
        std::getline(input, tmp);
        std::vector<string> res = predict(sentence);

        output << sentence << std::endl;
        for (int i = 0; i < res.size(); i++) {
            output  << res[i] << " ";
        }
        output << std::endl << std::endl;
        ind++;
        if (ind % 5 == 0) {
            std::cout << ind << std::endl;
        }
    }
    input.close();
    output.close();
}

void predict_for_php(int argc, char* argv[]) {
    std::string model = argv[2];
    Predictor* p_predictor = new Predictor(model);
    std::string input;
    std::string special_chars = ".,()\"'-:$/?;[]!";
    for (int i = 3; i < argc; i++) {
        bool ini_special = false;
        bool end_special = false;
        std::string original_word = argv[i];
        for (int j = 0; j < special_chars.size(); j++) {
            if (original_word[0] == special_chars[j]) {
                ini_special = true;
            }
            if (original_word[original_word.size() - 1] == special_chars[j]) {
                end_special = true;
            }
        }
        if (!ini_special && !end_special) {
            input += argv[i];
        }
        if (ini_special) {
            input += original_word[0] + " ";
            for (int j = 1; j < original_word.size(); j++) {
                input += original_word[j];
            }
        }
        if (end_special) {
            for (int j = 0; j < original_word.size() - 1; j++) {
                input += original_word[j];
            }
            input += " ";
            input += original_word[original_word.size() - 1];
        }
        if (i != argc - 1) {
            input += " ";
        }
    }
    std::vector<std::string> res = p_predictor->predict(input);
    std::string res_str;
    for (int i = 0; i < res.size(); i++) {
        res_str += res[i] + " ";
    }
    std::cout << input << std::endl << res_str << std::endl;

    delete p_predictor;
}

int main(int argc, char* argv[]) {
    // ./predictor mode model input
    // e.g. 
    // single test: ./predictor 0 ../../data/intermedia_data/model.data.baseline.8189
    // batch test: ./predictor 0 ../../data/intermedia_data/model.data.baseline.8189
    // train/testa/testb
    // php test: ./predictor php model_file input(I am from China.)
    std::string help_str = "./predictor mode model (input) \nsingle test: ./predictor 0 ../../data/intermedia_data/model.data.baseline.8189\nbatch test: ./predictor 1 ../../data/intermedia_data/model.data.baseline.8189 train/testa/testb";
    if (argc >= 4) {
        std::string mode = argv[1];
        if (mode == "php") {
            predict_for_php(argc, argv);
            return 0;
        }
    }
    if (argc == 3) {
        std::string mode = argv[1];
        if (mode != "0") {
            std::cout << help_str << std::endl;
            return 0;
        } 
        std::string model = argv[2];
        Predictor* p_predictor = new Predictor(model);
        while (true) {
            std::cout << "input:" << std::endl;
            std::string sentence;
            std::getline(std::cin, sentence);
            std::vector<std::string> res = p_predictor->predict(sentence);
            for (int i = 0; i < res.size(); i++) {
                std::cout << res[i] << " ";
            }
            std::cout << std::endl << std::endl;;
        }
    }
    if (argc == 4) {
        std::string mode = argv[1];
        if (mode != "1") {
            std::cout << help_str << std::endl;
            return 0;
        } 
        std::string model = argv[2];
        Predictor* p_predictor = new Predictor(model);
        std::string data_str = argv[3];
        std::string test_data_path = "../../data/ner/eng." + data_str + ".new";
        p_predictor->batch_predict(test_data_path);
    }
    std::cout << help_str << std::endl;
    return 0;
    //std::string a = "../../data/intermedia_data/model.data.baseline.8189"
    //p_predictor->batch_predict("../../data/ner/eng.testa.new");
    /*
    std::string sentence = "EU rejects German call to boycott British lamb .";

    std::cout << "input: " << std::endl;
    std::getline(std::cin, sentence);

    std::vector<std::string> res = p_predictor->predict(sentence);
    std::cout << sentence << std::endl;
    for (int i = 0; i < res.size(); i++) {
        std::cout << res[i] << " ";
    }
    std::cout << std::endl << std::endl;;
    */
}
