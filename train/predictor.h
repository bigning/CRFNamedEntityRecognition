#include "util.h"
#include "feature_extractor.h"
#include <string>
#include <iostream>
#include <vector>
#include <map>

using std::vector;

class Predictor {
public:
    Predictor(std::string& model);
    void load_model();
    std::vector<std::string> predict(std::string input);
    void cal_log_lambda(vector<vector<double> >& log_lambda);
    std::vector<std::string> decode(vector<vector<double> >& log_lambda);

    void batch_predict(const std::string& input_filename);

private:
    std::vector<double> weights_;
    std::string model_filename_;
    DataLabel data_label_;
    
    std::map<int, std::string> label_to_tag_;


    FeatureExtractor* p_feature_extractor_;
};
