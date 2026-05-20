#include "target_selector.h"
#include <algorithm>

void TargetSelector::init(const cv::Size& frame_size) {
    frame_size_ = frame_size;
}

FrameResult TargetSelector::update(const std::vector<ArmorObject>& detections) {
    FrameResult result;

    if (detections.empty()) return result;  // 无检测，返回全零

    // 按置信度从高到低排序
    auto sorted = detections;
    std::sort(sorted.begin(), sorted.end(),
              [](const ArmorObject& a, const ArmorObject& b) {
                  return a.confidence > b.confidence;  // 降序
              });

    // 最多取前 4 个（MAX_ARMOR_COUNT = 4）
    result.detected_count = std::min((int)sorted.size(), MAX_ARMOR_COUNT);

    for (int i = 0; i < result.detected_count; i++) {
        result.centers[i] = sorted[i].center;   // 填入中心点
    }
    // 不足 4 个的位置保持默认的 (0,0)

    (void)detections;

    FrameResult result;
    return result;
}
