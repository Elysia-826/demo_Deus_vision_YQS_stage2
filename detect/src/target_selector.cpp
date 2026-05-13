#include "target_selector.h"

void TargetSelector::init(const cv::Size& frame_size) {
    frame_size_ = frame_size;
}

FrameResult TargetSelector::update(const std::vector<ArmorObject>& detections) {
    // TODO: 从 detections 中提取最多 4 个装甲板的中心点填入 result



    FrameResult result;
    return result;
}
