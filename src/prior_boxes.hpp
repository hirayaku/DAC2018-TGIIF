#ifndef DEEPHI_PRIORBOXES_HPP_
#define DEEPHI_PRIORBOXES_HPP_

#include <vector>
#include <memory>
#include <utility>

namespace deephi {

class PriorBoxes {
 
 public:
  PriorBoxes(int image_width, int image_height, 
      int layer_width, int layer_height,
      const std::vector<float>& variances,
      const std::vector<float>& min_sizes, const std::vector<float>& max_sizes,
      const std::vector<float>& aspect_ratios, float offset,
      float step_width = 0.f, float step_height = 0.f,
      bool flip = true, bool clip = false);

  const std::vector<std::shared_ptr<std::vector<float> > >& priors() const {
    return priors_;
  }

 protected:
  
  void CreatePriors();

  std::vector<std::shared_ptr<std::vector<float> > > priors_;

  std::pair<int, int> image_dims_;
  std::pair<int, int> layer_dims_;
  std::pair<float, float> step_dims_;

  std::vector<std::pair<float, float> > boxes_dims_;

  float offset_;
  bool clip_;

  std::vector<float> variances_;
};

}

#endif
