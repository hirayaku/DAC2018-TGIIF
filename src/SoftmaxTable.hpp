#ifndef SOFTMAX_TABLE_HPP_
#define SOFTMAX_TABLE_HPP_

#include <tuple>
using namespace std;

class SoftmaxTable {
public:
    vector<double> exp_table_;
    vector<float> score_;
    float fix_scale_;
    int class_num_;
    int softmax_size_;

    SoftmaxTable(float fix_scale, int softmax_size, int class_num) {
        fix_scale_ = fix_scale;
        softmax_size_ = softmax_size;
        class_num_ = class_num;
        for (int i = 0; i < 256; i++) {
            float temp = i - 128;
            exp_table_.push_back(exp(temp * fix_scale_));
        }
        score_.reserve(softmax_size_ / class_num_);
        //cols_.reserve(softmax_size_ / class_num_);
    }

    //tuple<int, int, float> cal_softmax(int8_t* input) {
    tuple<int, float> cal_softmax(int8_t* input) {
        for (int i = 0; i < softmax_size_; i = i + class_num_) {
            double sum = 0;
            auto max_pt = max_element(input + i + 1, input + i + class_num_);

            for(int j = i; j < i + class_num_ / 4 * 4; j += 4) {
                sum += exp_table_[input[j] + 128] + exp_table_[input[j+1] + 128] +
                       exp_table_[input[j+2] + 128] + exp_table_[input[j+3] + 128];

            }
            for(int j = i + class_num_ / 4 * 4; j < class_num_; j++) {
                sum += exp_table_[input[j] + 128];
            }
            auto max_value = exp_table_[*max_pt + 128];

            score_.push_back(max_value / sum);
        }

        auto max_score_pt = max_element(score_.begin(), score_.end());
        auto max_ind = distance(score_.begin(), max_score_pt);
        score_.clear();
        return tuple<int, float>(max_ind, *max_score_pt);
    }


};

#endif
