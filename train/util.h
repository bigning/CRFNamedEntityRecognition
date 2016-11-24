#include <iostream>
#include <vector>

typedef std::vector<int> SparseVector;

double log_sum(double a, double b);
void log_sum_test();

double weight_time_feature(std::vector<double>& weights, SparseVector& feature);
