#pragma once

#include "target_selector.h"
#include <string>
#include <vector>

/**
 * @brief 评估模块
 *
 * 读取 assets/std.csv 作为 ground truth，与运行时输出逐帧对比
 * 精度: 高斯衰减将距离误差映射到 0-100
 * 速度: FPS 映射到 0-100
 * 综合 = 精度 * 0.7 + 速度 * 0.3
 */
class Evaluator {
public:
    Evaluator() = default;
    ~Evaluator() = default;

    bool init(const std::string& std_csv_path = "assets/std.csv", float sigma = 15.0f);
    void submit(int frame_id, const FrameResult& result);
    void printResult(float avg_fps);

private:
    struct GroundTruth {
        int frame_id;
        int count;
        std::array<cv::Point2f, MAX_ARMOR_COUNT> centers;
    };

    std::vector<GroundTruth> gt_data_;
    std::vector<float> frame_scores_;
    float sigma_ = 15.0f;
    bool loaded_ = false;
};
