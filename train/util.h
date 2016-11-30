#ifndef UTIL_H_
#define UTIL_H_
#include <iostream>
#include <vector>
#include <string>

using std::vector;
typedef std::vector<int> SparseVector;

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


double log_sum(double a, double b);
void log_sum_test();

double weight_time_feature(std::vector<double>& weights, SparseVector& feature);

void to_lowercase(std::string& input, std::string& output);
#endif
