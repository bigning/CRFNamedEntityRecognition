#include "util.h"

#include <vector>
#include <iostream>
#include <string>

struct DataLabel {
    std::vector<SparseVector> data;
    std::vector<int> label;
    DataLabel (std::vector<SparseVector> data_, std::vector<int> label_): 
        data(data_), label(label_) {}
    DataLabel() {}
};

class Trainer {
public:
    Trainer();
    void run();

    void load_train_data();
    void weight_initialization();

    void cal_gradients(int i, std::vector<double>& gradients);
    void cal_log_alpha(int i);
    void cal_log_beta(int i);
private:
    std::vector<DataLabel> train_data_;
    std::string train_data_file_;
    int train_num_;
    int feature_length_;

    // theta
    std::vector<double> weights_;
    std::vector<double> gradients_;
};
