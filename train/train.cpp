#include "train.h"

#include <fstream>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>

using std::vector;
using std::string;

Trainer::Trainer() {
    train_data_file_ = "../../data/ner/eng.train.new";
    train_num_ = 0;
    feature_length_ = 0;
    p_feature_extractor_ = new FeatureExtractor();
    feature_length_ = p_feature_extractor_->selected_feature_size();
    classes_ = p_feature_extractor_->label_map().size();

    train_num_ = 42123;

    // sgd parameter
    iter_ = 50;
    regularization_weight_ = 0;
    learning_rate_ = 0.02;

    // helper

    // debug
    sum_abs_weights = 0;
    sum_abs_grads = 0;
    time_ = 0;
}

Trainer::~Trainer() {
}

void Trainer::load_train_data() {
    int ind = 0;

    double time_1 = 0;
    double time_2 = 0;

    train_data_ = std::vector<DataLabel>(train_num_);

    std::ifstream train_data_file(train_data_file_.c_str());
    std::string data_line;
    std::string tag_line;
    std::string empty_line;
    std::clock_t begin = clock();
    while (std::getline(train_data_file, data_line)) {
        std::getline(train_data_file, tag_line);
        std::getline(train_data_file, empty_line);

        std::vector<std::string> data;
        std::vector<int> label;
        
        p_feature_extractor_->sentence2input_data(data_line, train_data_[ind].data); 

        p_feature_extractor_->tags2lable_data(tag_line, train_data_[ind].label);

        vector<vector<vector<SparseVector> > > features;
        p_feature_extractor_->extract_features(train_data_[ind].data,
                train_data_[ind].features);

        //train_data_[ind] = DataLabel(data, label, features);

        ind++;

        if (ind % 100 == 0) {
            std::clock_t end = clock();
            std::cout << "[INFO]: computing training feature: " << ind << " time: "
                << (double)(end - begin) / CLOCKS_PER_SEC << std::endl;
            begin = clock();
        }
    }
    train_num_ = ind;
    std::clock_t end = clock();
    std::cout << (double)(end - begin) / CLOCKS_PER_SEC << std::endl;
    train_data_file.close();
}

void Trainer::weight_initialization() {
    weights_ = std::vector<double>(feature_length_, 0.0);
    
    for (int i = 0; i < weights_.size(); i++) {
        weights_[i] = ((double)rand()) / RAND_MAX;
    }

    gradients_ = std::vector<double>(feature_length_, 0.0);
}

void Trainer::cal_log_alpha(int i, std::vector<std::vector<double> >& log_alpha) {
    DataLabel& data_label = train_data_[i];
    int T = data_label.label.size() - 2;
     log_alpha = std::vector<std::vector<double> >(data_label.data.size(),
            std::vector<double>(classes_, -1.0));

    // log_alpha_1_j
    for (int j = 0; j < classes_; j++) {
        SparseVector& feature = data_label.features[1][j][data_label.label[0]];
        log_alpha[1][j] = weight_time_feature(weights_, feature);
    }

    for (int t = 2; t <= T; t++) {
        for (int j = 0; j < classes_; j++) {
            double tmp_sum = 0;
            // i = 0
            // log(fai(j, i=0, xt)) + log(alpha_t-1_i). ref: notes P2
            SparseVector& tmp_feature = data_label.features[t][j][0];
            tmp_sum = weight_time_feature(weights_, tmp_feature) + 
                log_alpha[t - 1][0];
            for (int i = 1; i < classes_; i++) {
                double tmp_sum_2 = 0;
                SparseVector& tmp_feature = data_label.features[t][j][i];
                tmp_sum_2 = weight_time_feature(weights_, tmp_feature) + 
                    log_alpha[t - 1][i];
                tmp_sum = log_sum(tmp_sum, tmp_sum_2);
            }
            log_alpha[t][j] = tmp_sum;
        }
    }
}

void Trainer::cal_log_beta(int i, std::vector<std::vector<double> >& log_beta) {
    DataLabel& data_label = train_data_[i];
    int T = data_label.label.size() - 2;

    log_beta = std::vector<std::vector<double> >(data_label.data.size(), 
            std::vector<double>(classes_, -1.0));
    for (int i = 0; i < classes_; i++) {
        log_beta[T][i] = 0;
    }

    for (int t = T - 1; t >= 0; t--) {
        for (int i = 0; i < classes_; i++) {
            double tmp_sum = 0;
            // j = 0;
            // log(fai(j=0, i, xt) + log_beta_t+1_j)
            SparseVector& tmp_feature = data_label.features[t+1][0][i];
            tmp_sum = weight_time_feature(weights_, tmp_feature) +
                log_beta[t+1][0];
            for (int j = 1; j < classes_; j++) {
                double tmp_sum_2 = 0;
                SparseVector& tmp_feature = data_label.features[t+1][j][i];
                tmp_sum_2 = weight_time_feature(weights_, tmp_feature) + 
                    log_beta[t+1][j];
                tmp_sum = log_sum(tmp_sum, tmp_sum_2);
            }
            log_beta[t][i] = tmp_sum;
        }
    }
}

void Trainer::train() {
    int print_every = 500;
    double sum_likelihood = 0;
    int num = 0;
    std::vector<int> rnd_shuffle(train_num_, 0);
    for (int i = 0; i < train_num_; i++) {
        rnd_shuffle[i] = i;
    }

    for (int i = 0; i < iter_; i++) {
        std::random_shuffle(rnd_shuffle.begin(), rnd_shuffle.end());
        for (int j = 0; j < train_num_; j++) {
            double likelihood = cal_gradients(rnd_shuffle[j]);
            update_weights();

            sum_likelihood += likelihood;
            
            num++; 

            if (num % print_every == 0) {

                std::cout << "[INFO]: iter " << i << ", avg likelihood: " << 
                    sum_likelihood / (double)print_every << std::endl;
                std::cout << "[INFO]: iter " << i << ", grad/weight: " << sum_abs_grads/sum_abs_weights << std::endl;
                sum_likelihood = 0;
                sum_abs_grads = 0;
                sum_abs_weights = 0;

            }
        }
        // update learning rate
        learning_rate_ *= 0.88;
        save_model(i);
    }
}

void Trainer::save_model(int i) {
    std::ofstream out("../../data/intermedia_data/model.data");
    out << i << std::endl;
    for (int i = 0; i < weights_.size(); i++) {
        out << weights_[i] << " ";
    }
    out.close();
}

double Trainer::cal_gradients(int i) {
   DataLabel& data_label = train_data_[i];
   int T = data_label.data.size() - 2;

   // compute log_z
   vector<vector<double> > log_alpha;
   cal_log_alpha(i, log_alpha);
   vector<vector<double> > log_beta;
   cal_log_beta(i, log_beta);

   double log_z = 0;
   /*
   double log_z_alpha = log_alpha[T][0];
   for (int i = 1; i < classes_; i++) {
       log_z_alpha = log_sum(log_z_alpha, log_alpha[T][i]);
   }
   */
   double log_z_beta = (log_beta[0][0]);
   log_z = log_z_beta;

   // compute P_t(y_t, y_t-1,x) P316
   vector<vector<vector<double> > > p(data_label.data.size(), vector<vector<double> >(classes_,
               vector<double>(classes_, 0)));
   //special case for P_1(y_1, y_0, x), only when y_0 = 0, this value is not zero
   for (i = 0; i < classes_; i++) {
       SparseVector& sparse_feature = data_label.features[1][i][0];
       double tmp1 = weight_time_feature(weights_, sparse_feature);
       double log_p_1_i_0 = tmp1 + log_beta[1][i] - log_z;
       p[1][i][0] = exp(log_p_1_i_0);
   }
   for (int t = 2; t <= T; t++) {
       for (int i = 0; i < classes_; i++) {
           for (int j = 0; j < classes_; j++) {
               SparseVector& sparse_feature = data_label.features[t][i][j];
               double log_fai = weight_time_feature(weights_, sparse_feature);
               double log_p = log_alpha[t - 1][j] + log_fai + log_beta[t][i] - log_z;
               p[t][i][j] = exp(log_p); 
           }
       }
   }

   // cal gradients
   double avg_gradients = 0;
   std::fill(gradients_.begin(), gradients_.end(), 0);
   for (int k = 0; k < weights_.size(); k++) {
       double tmp = 0;
       for (int t = 1; t <= T; t++) {
           // f_k(y_t, y_t-1, x)
           bool is_kth_nonzero = false;
           SparseVector& sparse_feature =
               data_label.features[t][data_label.label[t]][data_label.label[t - 1]];
           for (int i = 0; i < sparse_feature.size(); i++) {
               if (sparse_feature[i] == k) {
                   is_kth_nonzero = true;
                   break;
               }
           }
           if (is_kth_nonzero) {
               tmp += 1;
           }
           
           for (int i = 0; i < classes_; i++) {
               for (int j = 0; j < classes_; j++) {
                   SparseVector& sparse_feature= data_label.features[t][i][j];
                   bool is_kth_nonzero = false;
                   for (int m = 0; m < sparse_feature.size(); m++) {
                       if (sparse_feature[m] == k) {
                           is_kth_nonzero = true;
                           break;
                       }
                   }
                   if (!is_kth_nonzero) {
                       continue;
                   }
                   tmp -= p[t][i][j];
               }
           }
       }
       tmp -= 2 * regularization_weight_ * weights_[k];
       gradients_[k] = tmp;
       avg_gradients += tmp;
   }
   avg_gradients /= (double)(weights_.size());

   // cal likelyhood
   double l = 0;
   for (int t = 1; t <= T; t++) {
       SparseVector& feature = data_label.features[t][data_label.label[t]][data_label.label[t - 1]];
       l += weight_time_feature(weights_, feature);
   }
   l -= log_z;
   return l;
}

void Trainer::update_weights() {
    for (int i = 0; i < weights_.size(); i++) {
        weights_[i] += learning_rate_ * gradients_[i];
        if (gradients_[i] > 0.00001 || gradients_[i] < -0.00001) {
            sum_abs_weights += abs(weights_[i]);
            sum_abs_grads += abs(gradients_[i]);
        }
    }
}

void Trainer::run() {
    load_train_data();
    std::cout << "[INFO]: load " << train_num_ << " train samples. Feature dimension: "
        << feature_length_ << std::endl << std::endl;;

    weight_initialization();
    std::cout << "[INFO]: weights are randomly initialized!" << std::endl << std::endl;

    train();
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
