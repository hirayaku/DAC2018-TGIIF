#include "ssd_detector.hpp"

#include <cmath>
#include <algorithm>
#include <functional>
#include <tuple>
#include <thread>
#include <iostream>
#include <opencv2/core/core.hpp>

#include "time_helper.hpp"

namespace deephi {

using namespace cv;
using namespace std;

SSDdetector::SSDdetector(unsigned int num_classes,  // int background_label_id,
                         CodeType code_type, bool variance_encoded_in_target,
                         unsigned int keep_top_k,
                         const vector<float>& confidence_threshold,
                         unsigned int nms_top_k, float nms_threshold, float eta,
                         const vector<shared_ptr<vector<float>>>& priors,
                         float scale, bool clip)
    : num_classes_(num_classes),
      // background_label_id_(background_label_id),
      code_type_(code_type),
      variance_encoded_in_target_(variance_encoded_in_target),
      keep_top_k_(keep_top_k),
      confidence_threshold_(confidence_threshold),
      nms_top_k_(nms_top_k),
      nms_threshold_(nms_threshold),
      eta_(eta),
      priors_(priors),
      scale_(scale),
      clip_(clip) {
  num_priors_ = priors_.size();
  nms_confidence_ = *std::min_element(
      confidence_threshold_.begin() + 1, confidence_threshold_.end());
}

template <typename T>
void SSDdetector::Detect(const T* loc_data, const float* conf_data,
                         MultiDetObjects* result, int location, int classification) {
  decoded_bboxes_.clear();
  const T(*bboxes)[4] = (const T(*)[4])loc_data;

//  __TIC__(Sort);
  int num_det = 0;
  vector<vector<int>> indices(num_classes_);
  vector<vector<pair<float, int>>> score_index_vec(num_classes_);

#if 0
  // Get top_k scores (with corresponding indices).
  GetMultiClassMaxScoreIndexMT(conf_data, 1, num_classes_ - 1,
                               &score_index_vec);
#else
/*
  int location = -1;
  float conf = 0;
  int classification = -1;
  for(int i = 0; i < num_priors_; ++i) {
    for(int j = 1; j < num_classes_; ++j) {
      int offset = i * num_classes_ + j;
      if (conf_data[offset] > confidence_threshold_[i] && conf_data[offset] > conf) {
        conf = conf_data[offset];
        location = i;
        classification = j;
      }
    }
  }
  score_index_vec[0].push_back(make_pair(conf, location));
*/

  float conf = conf_data[location*num_classes_+classification];
  if(conf > confidence_threshold_[classification]) {
    score_index_vec[0].push_back(make_pair(conf, location));
    ApplyOneClassNMS(bboxes, conf_data, classification, score_index_vec[0], &(indices[0]));
    num_det = 1;
  }
#endif
  // for (int c = 1; c < num_classes_; ++c) {
  // if (c == background_label_id_) {
  //   continue;
  // }
  // GetOneClassMaxScoreIndex(conf_data, c, &score_index_vec[c]);
  // }

//  __TOC__(Sort);
//  __TIC__(NMS);
#if 0
  for (int c = 1; c < num_classes_; ++c) {
    // Perform NMS for one class
    ApplyOneClassNMS(bboxes, conf_data, c, score_index_vec[c], &(indices[c]));

    num_det += indices[c].size();
  }
#endif

  if (keep_top_k_ > 0 && num_det > keep_top_k_) {
    vector<tuple<float, int, int>> score_index_tuples;
    for (auto label = 0; label < num_classes_; ++label) {
      const vector<int>& label_indices = indices[label];
      for (auto j = 0; j < label_indices.size(); ++j) {
        auto idx = label_indices[j];
        auto score = conf_data[idx * num_classes_ + label];
        score_index_tuples.emplace_back(score, label, idx);
      }
    }

    // Keep top k results per image.
    std::sort(score_index_tuples.begin(), score_index_tuples.end(),
              [](const tuple<float, int, int>& lhs,
                 const tuple<float, int, int>& rhs) {
                return get<0>(lhs) > get<0>(rhs);
              });
    score_index_tuples.resize(keep_top_k_);

    indices.clear();
    indices.resize(num_classes_);
    for (auto& item : score_index_tuples) {
      indices[get<1>(item)].push_back(get<2>(item));
    }

    num_det = keep_top_k_;
  }

//  __TOC__(NMS);

//  __TIC__(Box);
/*
  for (auto label = 1; label < indices.size(); ++label) {
    for (auto idx : indices[label]) {
      auto score = conf_data[idx * num_classes_ + label];
      if (score < confidence_threshold_[label]) {
        continue;
      }
      auto& bbox = decoded_bboxes_[idx];
	    bbox[0] = std::max(std::min(bbox[0], 1.f), 0.f);
	    bbox[1] = std::max(std::min(bbox[1], 1.f), 0.f);
	    bbox[2] = std::max(std::min(bbox[2], 1.f), 0.f);
	    bbox[3] = std::max(std::min(bbox[3], 1.f), 0.f);
cout << "index: " << idx << endl;
cout << "score: " << score << endl;
cout << "lol: " << bbox[0] << " " << bbox[1] << " " << bbox[2] << " " << bbox[3] << endl;
	    auto box_rect = Rect_<float>(Point2f(bbox[0], bbox[1]),
          Point2f(bbox[2], bbox[3]));
      result->emplace_back(label, score, box_rect);
    }
  }
*/
  if(conf > confidence_threshold_[classification]) {
    auto& bbox = decoded_bboxes_[location];
    bbox[0] = std::max(std::min(bbox[0], 1.f), 0.f);
    bbox[1] = std::max(std::min(bbox[1], 1.f), 0.f);
    bbox[2] = std::max(std::min(bbox[2], 1.f), 0.f);
    bbox[3] = std::max(std::min(bbox[3], 1.f), 0.f);
    auto box_rect = Rect_<float>(Point2f(bbox[0], bbox[1]),
          Point2f(bbox[2], bbox[3]));
    result->emplace_back(classification, conf, box_rect);
  }

//  __TOC__(Box);
}

template void SSDdetector::Detect(
    const int* loc_data, const float* conf_data,
    MultiDetObjects* result, int, int);
template void SSDdetector::Detect(
    const int8_t* loc_data, const float* conf_data,
    MultiDetObjects* result, int, int);

template <typename T>
void SSDdetector::ApplyOneClassNMS(
    const T (*bboxes)[4], const float* conf_data, int label,
    const vector<pair<float, int>>& score_index_vec, vector<int>* indices) {
  // Get top_k scores (with corresponding indices).
  // vector<pair<float, int> > score_index_vec;
  // GetOneClassMaxScoreIndex(conf_data, label, &score_index_vec);

  // Do nms.
  float adaptive_threshold = nms_threshold_;
  indices->clear();
  int i = 0;
  while (i < score_index_vec.size()) {
//	__TIC__(Decode)
    const int idx = score_index_vec[i].second;
    if (decoded_bboxes_.find(idx) == decoded_bboxes_.end()) {
      DecodeBBox(bboxes, idx, true);
    }
//	__TOC__(Decode)
//	__TIC__(OVERLAP)
    bool keep = true;
    for (int k = 0; k < indices->size(); ++k) {
      if (keep) {
        const int kept_idx = (*indices)[k];
        float overlap = JaccardOverlap(bboxes, idx, kept_idx);
        keep = overlap <= adaptive_threshold;
      } else {
        break;
      }
    }
    if (keep) {
      indices->push_back(idx);
    }
    ++i;
    if (keep && eta_ < 1 && adaptive_threshold > 0.5) {
      adaptive_threshold *= eta_;
    }
//	__TOC__(OVERLAP)
  }
}

template void SSDdetector::ApplyOneClassNMS(
    const int (*bboxes)[4], const float* conf_data, int label,
    const vector<pair<float, int>>& score_index_vec, vector<int>* indices);
template void SSDdetector::ApplyOneClassNMS(
    const int8_t (*bboxes)[4], const float* conf_data, int label,
    const vector<pair<float, int>>& score_index_vec, vector<int>* indices);

void SSDdetector::GetOneClassMaxScoreIndex(
    const float* conf_data, int label,
    vector<pair<float, int>>* score_index_vec) {
  //__TIC__(PUSH2)
  conf_data += label;
  for (int i = 0; i < num_priors_; ++i) {
    auto score = *conf_data;
    if (score > nms_confidence_) {
      score_index_vec->emplace_back(score, i);
    }
    conf_data += num_classes_;
  }
  //__TOC__(PUSH2)
  //__TIC__(SORT2)
  std::stable_sort(
      score_index_vec->begin(), score_index_vec->end(),
      [](const pair<float, int>& lhs, const pair<float, int>& rhs) {
        return lhs.first > rhs.first;
      });
  //__TOC__(SORT2)

  if (nms_top_k_ > -1 && nms_top_k_ < score_index_vec->size()) {
    score_index_vec->resize(nms_top_k_);
  }
}

void SSDdetector::GetMultiClassMaxScoreIndex(
    const float* conf_data, int start_label, int num_classes,
    vector<vector<pair<float, int>>>* score_index_vec) {
  for (auto i = start_label; i < start_label + num_classes; ++i) {
    GetOneClassMaxScoreIndex(conf_data, i, &((*score_index_vec)[i]));
  }
}

void SSDdetector::GetMultiClassMaxScoreIndexMT(
    const float* conf_data, int start_label, int num_classes,
    vector<vector<pair<float, int>>>* score_index_vec, int threads) {
  // CHECK_GT(threads, 0);
  int thread_classes = num_classes / threads;
  int last_thread_classes = num_classes % threads + thread_classes;

  vector<std::thread> workers;

  auto c = start_label;
  for (auto i = 0; i < threads - 1; ++i) {
    workers.emplace_back(&SSDdetector::GetMultiClassMaxScoreIndex, this,
                         conf_data, c, thread_classes, score_index_vec);
    c += thread_classes;
  }
  workers.emplace_back(&SSDdetector::GetMultiClassMaxScoreIndex, this,
                       conf_data, c, last_thread_classes, score_index_vec);

  for (auto& worker : workers)
    if (worker.joinable()) worker.join();
}

void BBoxSize(vector<float>& bbox, bool normalized) {
  float width = bbox[2] - bbox[0];
  float height = bbox[3] - bbox[1];
  if (width > 0 && height > 0) {
    if (normalized) {
      bbox[4] = width * height;
    } else {
      bbox[4] = (width + 1) * (height + 1);
    }
  } else {
    bbox[4] = 0.f;
  }
}

float IntersectBBoxSize(const vector<float>& bbox1, const vector<float>& bbox2,
                        bool normalized) {
  if (bbox2[0] > bbox1[2] || bbox2[2] < bbox1[0] || bbox2[1] > bbox1[3] ||
      bbox2[3] < bbox1[1]) {
    // Return 0 if there is no intersection.
    return 0.f;
  }

  vector<float> intersect_bbox(5);
  intersect_bbox[0] = max(bbox1[0], bbox2[0]);
  intersect_bbox[1] = max(bbox1[1], bbox2[1]);
  intersect_bbox[2] = min(bbox1[2], bbox2[2]);
  intersect_bbox[3] = min(bbox1[3], bbox2[3]);
  BBoxSize(intersect_bbox, normalized);
  return intersect_bbox[4];
}

/*
void ClipBBox(const NormalizedBBox& bbox, NormalizedBBox* clip_bbox) {
  clip_bbox->set_xmin(std::max(std::min(bbox.xmin(), 1.f), 0.f));
  clip_bbox->set_ymin(std::max(std::min(bbox.ymin(), 1.f), 0.f));
  clip_bbox->set_xmax(std::max(std::min(bbox.xmax(), 1.f), 0.f));
  clip_bbox->set_ymax(std::max(std::min(bbox.ymax(), 1.f), 0.f));
  clip_bbox->clear_size();
  clip_bbox->set_size(BBoxSize(*clip_bbox));
  clip_bbox->set_difficult(bbox.difficult());
}

void ClipBBox(const NormalizedBBox& bbox, const float height, const float width,
              NormalizedBBox* clip_bbox) {
  clip_bbox->set_xmin(std::max(std::min(bbox.xmin(), width), 0.f));
  clip_bbox->set_ymin(std::max(std::min(bbox.ymin(), height), 0.f));
  clip_bbox->set_xmax(std::max(std::min(bbox.xmax(), width), 0.f));
  clip_bbox->set_ymax(std::max(std::min(bbox.ymax(), height), 0.f));
  clip_bbox->clear_size();
  clip_bbox->set_size(BBoxSize(*clip_bbox));
  clip_bbox->set_difficult(bbox.difficult());
}
*/

template <typename T>
float SSDdetector::JaccardOverlap(const T (*bboxes)[4], int idx, int kept_idx,
                                  bool normalized) {
  /*
    if (decoded_bboxes_.find(idx) == decoded_bboxes_.end()) {
      DecodeBBox(bboxes, idx, normalized);
    }
    if (decoded_bboxes_.find(kept_idx) == decoded_bboxes_.end()) {
      DecodeBBox(bboxes, kept_idx, normalized);
    }
  */
  const vector<float>& bbox1 = decoded_bboxes_[idx];
  const vector<float>& bbox2 = decoded_bboxes_[kept_idx];
  float intersect_size = IntersectBBoxSize(bbox1, bbox2, normalized);
  return intersect_size <= 0 ? 0 : intersect_size /
                                       (bbox1[4] + bbox2[4] - intersect_size);
}

template float SSDdetector::JaccardOverlap(const int (*bboxes)[4], int idx,
                                           int kept_idx, bool normalized);
template float SSDdetector::JaccardOverlap(const int8_t (*bboxes)[4], int idx,
                                           int kept_idx, bool normalized);

template <typename T>
void SSDdetector::DecodeBBox(const T (*bboxes)[4], int idx, bool normalized) {
  vector<float> bbox(5, 0);
  // scale bboxes
  transform(bboxes[idx], bboxes[idx] + 4, bbox.begin(),
            std::bind2nd(multiplies<float>(), scale_));
  for (int i =0; i < 1; i++) {
	// LOG(INFO) << "lalalal========" << bbox[0] << " " << bbox[1] << " " << bbox[2] << " " << bbox[3] << " " << bbox[4];
  }
  auto& prior_bbox = priors_[idx];

  if (code_type_ == CodeType::CORNER) {
    if (variance_encoded_in_target_) {
      // variance is encoded in target, we simply need to add the offset
      // predictions.
      transform(bbox.begin(), bbox.end(), prior_bbox->begin(), bbox.begin(),
                plus<float>());
    } else {
      // variance is encoded in bbox, we need to scale the offset accordingly.
      transform(bbox.begin(), bbox.end(), prior_bbox->begin() + 4, bbox.begin(),
                multiplies<float>());
      transform(bbox.begin(), bbox.end(), prior_bbox->begin(), bbox.begin(),
                plus<float>());
    }
  } else if (code_type_ == CodeType::CENTER_SIZE) {
    float decode_bbox_center_x, decode_bbox_center_y;
    float decode_bbox_width, decode_bbox_height;
    if (variance_encoded_in_target_) {
      // variance is encoded in target, we simply need to retore the offset
      // predictions.
      decode_bbox_center_x = bbox[0] * (*prior_bbox)[10] + (*prior_bbox)[8];
      decode_bbox_center_y = bbox[1] * (*prior_bbox)[11] + (*prior_bbox)[9];
      decode_bbox_width = exp(bbox[2]) * (*prior_bbox)[10];
      decode_bbox_height = exp(bbox[3]) * (*prior_bbox)[11];
    } else {
      // variance is encoded in bbox, we need to scale the offset accordingly.
      decode_bbox_center_x =
          (*prior_bbox)[4] * bbox[0] * (*prior_bbox)[10] + (*prior_bbox)[8];
      decode_bbox_center_y =
          (*prior_bbox)[5] * bbox[1] * (*prior_bbox)[11] + (*prior_bbox)[9];
      decode_bbox_width = exp((*prior_bbox)[6] * bbox[2]) * (*prior_bbox)[10];
      decode_bbox_height = exp((*prior_bbox)[7] * bbox[3]) * (*prior_bbox)[11];
    }

    bbox[0] = decode_bbox_center_x - decode_bbox_width / 2.;
    bbox[1] = decode_bbox_center_y - decode_bbox_height / 2.;
    bbox[2] = decode_bbox_center_x + decode_bbox_width / 2.;
    bbox[3] = decode_bbox_center_y + decode_bbox_height / 2.;
  } else if (code_type_ == CodeType::CORNER_SIZE) {
    if (variance_encoded_in_target_) {
      // variance is encoded in target, we simply need to add the offset
      // predictions.
      bbox[0] *= (*prior_bbox)[10];
      bbox[1] *= (*prior_bbox)[11];
      bbox[2] *= (*prior_bbox)[10];
      bbox[3] *= (*prior_bbox)[11];
      transform(bbox.begin(), bbox.end(), prior_bbox->begin(), bbox.begin(),
                plus<float>());
    } else {
      // variance is encoded in bbox, we need to scale the offset accordingly.
      bbox[0] *= (*prior_bbox)[10];
      bbox[1] *= (*prior_bbox)[11];
      bbox[2] *= (*prior_bbox)[10];
      bbox[3] *= (*prior_bbox)[11];
      transform(bbox.begin(), bbox.end(), prior_bbox->begin() + 4, bbox.begin(),
                multiplies<float>());
      transform(bbox.begin(), bbox.end(), prior_bbox->begin(), bbox.begin(),
                plus<float>());
    }
  } else {
    // LOG(FATAL) << "Unknown LocLossType.";
  }

  BBoxSize(bbox, normalized);
  // if (clip_) {
  //   ClipBBox(*bbox, normalized);
  // }

  decoded_bboxes_.emplace(idx, std::move(bbox));
}

template void SSDdetector::DecodeBBox(const int (*bboxes)[4], int idx,
                                      bool normalized);
template void SSDdetector::DecodeBBox(const int8_t (*bboxes)[4], int idx,
                                      bool normalized);
}
