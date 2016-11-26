#include "feature_extractor.h"
#include <iostream>

int main() {
    FeatureExtractor* p_feature_extractor = new FeatureExtractor();
    //p_feature_extractor->test_extract_feature();
    p_feature_extractor->get_selected_feature_index();
    return 0;
}
