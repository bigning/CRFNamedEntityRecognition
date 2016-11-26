#include "feature_extractor.h"
#include "util.h"

#include <vector>
#include <iostream>
#include <string>
using std::vector;

struct DataLabel {
    std::vector<std::string> data;
    std::vector<int> label;
    //std::vector<SparseVector> feature;
    // features[t][i][j] = p_feature_extractor->extract_feature(t, i, j, data)
    vector<vector<vector<SparseVector> > > features;
    DataLabel (std::vector<std::string> data_, std::vector<int> label_, 
            vector<vector<vector<SparseVector> > >& features_):
        data(data_), label(label_), features(features_) {}
    DataLabel() {}
};

class Trainer {
public:
    Trainer();
    ~Trainer();
    void run();
    void train();

    void load_train_data();
    void weight_initialization();

    void extract_features(vector<std::string>& data,
            vector<vector<vector<SparseVector> > >& features);
    void cal_log_alpha(int i, std::vector<std::vector<double> >& log_alpha);
    void cal_log_beta(int i, std::vector<std::vector<double> >& log_beta);
private:
    // compute gradient, return likelihood
    double cal_gradients(int i);
    void update_weights();

    std::vector<DataLabel> train_data_;
    std::string train_data_file_;
    int train_num_;
    int feature_length_;
    int classes_;

    // theta
    std::vector<double> weights_;
    std::vector<double> gradients_;

    FeatureExtractor* p_feature_extractor_;

    // sgd parameters
    int iter_;
    double regularization_weight_;
    double learning_rate_;

    // helper
    double* original_feature_;

    // debug
    double sum_abs_weights;
    double sum_abs_grads;
};
