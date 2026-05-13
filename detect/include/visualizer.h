#pragma once

#include "detector.h"
#include "target_selector.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

/**
 * @brief 可视化模块
 *
 * 目标: 在画面上绘制检测结果与中心点
 */
class Visualizer {
public:
    void drawDetections(cv::Mat& frame, const std::vector<ArmorObject>& detections);
    void drawCenters(cv::Mat& frame, const FrameResult& result);
    void drawHUD(cv::Mat& frame, float fps, int detected_count);
};
