#ifndef DEEPHI_SSD_DETECTOR_HPP_
#define DEEPHI_SSD_DETECTOR_HPP_

#include <vector>
#include <memory>
#include <map>
#include <utility>
//#include "base/data.hpp"
#include <tuple>
#include <opencv2/core.hpp>

namespace deephi {

using SingleDetObject = std::tuple<int, float, cv::Rect_<float> >;
using MultiDetObjects = std::vector<SingleDetObject>;

class SSDdetector {

 public:
  enum CodeType { CORNER, CENTER_SIZE, CORNER_SIZE };

  SSDdetector(unsigned int num_classes,// int background_label_id,
      CodeType code_type, bool variance_encoded_in_target,
      unsigned int keep_top_k,
      const std::vector<float>& confidence_threshold,
      unsigned int nms_top_k, float nms_threshold, float eta,
      const std::vector<std::shared_ptr<std::vector<float> > >& priors,
      float scale = 1.f, bool clip = false);

  template <typename T>
  void Detect(const T* loc_data, const float* conf_data,
              MultiDetObjects* result, int, int);

  unsigned int num_classes() const { return num_classes_; }
  unsigned int num_priors() const { return priors_.size(); }

 protected:

  template <typename T>
  void ApplyOneClassNMS(const T (*bboxes)[4], const float* conf_data,
      int label, const std::vector<std::pair<float, int> >& score_index_vec,
      std::vector<int>* indices);

  void GetOneClassMaxScoreIndex(const float* conf_data, int label,
      std::vector<std::pair<float, int> >* score_index_vec);

  void GetMultiClassMaxScoreIndex(const float* conf_data,
      int start_label, int num_classes,
      std::vector<std::vector<std::pair<float, int> > >* score_index_vec);

  void GetMultiClassMaxScoreIndexMT(const float* conf_data,
      int start_label, int num_classes,
      std::vector<std::vector<std::pair<float, int> > >* score_index_vec,
      int threads = 2);

  template <typename T>
  float JaccardOverlap(const T (*bboxes)[4], int idx, int kept_idx,
      bool normalized = true);

  template <typename T>
  void DecodeBBox(const T (*bboxes)[4], int idx, bool normalized);

  std::map<int, std::vector<float> > decoded_bboxes_;

  const unsigned int num_classes_;
  // int background_label_id_;
  CodeType code_type_;
  bool variance_encoded_in_target_;
  unsigned int keep_top_k_;
  std::vector<float> confidence_threshold_;
  float nms_confidence_;
  unsigned int nms_top_k_;
  float nms_threshold_;
  float eta_;

  const std::vector<std::shared_ptr<std::vector<float> > >& priors_;
  float scale_;

  bool clip_;

  int num_priors_;
};

}

#endif
