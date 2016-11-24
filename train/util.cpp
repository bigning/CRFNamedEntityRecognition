#include "util.h"
#include <math.h>

double log_sum(double a, double b) {
    if (a < b) {
        return b + log(1 + exp(a - b));
    }
    return a + log(1 + exp(b - a));
}

void log_sum_test() {
    double a = 0.0000034;
    double b = 0.00234;
    double res_true = log(exp(a) + exp(b));
    double res_test = log_sum(a, b);
    std::cout << "true: " << res_true << "  test: " << res_test << std::endl;
}

double weight_time_feature(std::vector<double>& weights, SparseVector& feature) {
    double res = 0;
    for (int i = 0; i < feature.size(); i++) {
        res += weights[feature[i]];
    }
    return res;
}

void to_lowercase(std::string& input, std::string& output) {
    for (int i = 0; i < input.size(); i++) {
        if (input[i] >= 'A' && input[i] <= 'Z') {
            output[i] = input[i] + 32;
        } else {
            output[i] = input[i];
        }
    }
}
