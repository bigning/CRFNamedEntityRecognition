#include "util.h"

#include <iostream>
#include <iostream>
#include <map>
#include <vector>

using namespace std;

class FeatureExtractor {
public:
    FeatureExtractor();
    void load_words();
    void load_selected_feature_index();
    void extract_original_feature(int t, int y, int y_prev, vector<string>& input);
    // 1. feature selection 2. convert to sparse vector
    vector<int> extract_feature(int t, int y, int y_prev, vector<string>& input);
    void extract_features(vector<std::string>& data,
            vector<vector<vector<SparseVector> > >& features);
    void test_extract_feature();

    /*
     * extract feature from training data and label, 
     * so that, we can know the # of occurance for each feature,
     * then select the feature, that occures at lease twice
     * ref: NER with a Maximum Entropy Approach
     */
    void get_selected_feature_index();
    /* 
     * sentence: How are you ? -> <start> how are you ?
    */
    void sentence2input_data(std::string& sentence, vector<string>& data);
    /*
     * append "O" label to start
     * */
    void tags2lable_data(std::string& tags, vector<int>& label);

    int selected_feature_size() {return selected_feature_size_;}
    const std::map<std::string, int>& label_map() { return label_map_;}
    int label_num() { return label_num_; }

private:
    std::string train_data_filename_; 
    std::string lexicon_filename_;
    int label_num_;

    std::vector<int> selected_feature_ind_;
    int selected_feature_size_;
    std::string selected_feature_ind_save_filename;

    // O->0 B-PER->1 I-PER->2, ...
    std::map<std::string, int> label_map_; 

    // "word"->index
    std::map<std::string, int> word2ind_;
    int dict_size_;
    
    int* original_feature_;
    int original_feature_size_;
    int non_zero_num_;
    //
};
