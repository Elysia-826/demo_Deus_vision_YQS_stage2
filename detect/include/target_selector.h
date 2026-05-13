#pragma once

#include "detector.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <array>

/// 最多输出 4 个装甲板中心点
static constexpr int MAX_ARMOR_COUNT = 4;

/// 一帧的锁定结果: 最多 4 个中心点，不足的填 (0,0)
struct FrameResult {
    int detected_count = 0;
    std::array<cv::Point2f, MAX_ARMOR_COUNT> centers = {
        cv::Point2f(0, 0), cv::Point2f(0, 0),
        cv::Point2f(0, 0), cv::Point2f(0, 0)
    };
};

/**
 * @brief 目标锁定模块
 *
 * 目标: 从检测结果中提取每个装甲板的中心点坐标 (最多 4 个)
 */
class TargetSelector {
public:
    TargetSelector() = default;
    ~TargetSelector() = default;

    void init(const cv::Size& frame_size);

    FrameResult update(const std::vector<ArmorObject>& detections);

private:
    cv::Size frame_size_;
};
