#include "evaluate.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

bool Evaluator::init(const std::string& std_csv_path, float sigma) {
    sigma_ = sigma;

    std::ifstream file(std_csv_path);
    if (!file.is_open()) {
        std::cerr << "[Evaluate] Cannot open: " << std_csv_path << std::endl;
        return false;
    }

    std::string line;
    std::getline(file, line); // 跳过表头

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        GroundTruth gt;

        std::getline(ss, token, ','); gt.frame_id = std::stoi(token);
        std::getline(ss, token, ','); gt.count = std::stoi(token);

        for (int i = 0; i < MAX_ARMOR_COUNT; i++) {
            std::getline(ss, token, ','); gt.centers[i].x = std::stof(token);
            std::getline(ss, token, ','); gt.centers[i].y = std::stof(token);
        }

        gt_data_.push_back(gt);
    }

    file.close();
    loaded_ = true;
    std::cout << "[Evaluate] Loaded " << gt_data_.size() << " frames from " << std_csv_path << std::endl;
    return true;
}

void Evaluator::submit(int frame_id, const FrameResult& result) {
    if (!loaded_) return;

    const GroundTruth* gt = nullptr;
    for (const auto& g : gt_data_) {
        if (g.frame_id == frame_id) {
            gt = &g;
            break;
        }
    }
    if (!gt) return;

    if (gt->count == 0) {
        float frame_score = (result.detected_count == 0) ? 100.0f : 0.0f;
        frame_scores_.push_back(frame_score);
        return;
    }

    std::vector<bool> used(MAX_ARMOR_COUNT, false);
    float score_sum = 0.0f;

    for (int i = 0; i < gt->count && i < MAX_ARMOR_COUNT; i++) {
        float best_dist = 1e9f;
        int best_j = -1;

        for (int j = 0; j < result.detected_count && j < MAX_ARMOR_COUNT; j++) {
            if (used[j]) continue;
            float dx = result.centers[j].x - gt->centers[i].x;
            float dy = result.centers[j].y - gt->centers[i].y;
            float dist = std::sqrt(dx * dx + dy * dy);
            if (dist < best_dist) {
                best_dist = dist;
                best_j = j;
            }
        }

        float point_score = 0.0f;
        if (best_j >= 0) {
            point_score = 100.0f * std::exp(-(best_dist * best_dist) / (2.0f * sigma_ * sigma_));
            used[best_j] = true;
        }
        score_sum += point_score;
    }

    float frame_score = score_sum / static_cast<float>(gt->count);
    frame_scores_.push_back(frame_score);
}

void Evaluator::printResult(float avg_fps) {
    if (!loaded_) {
        std::cout << "[Evaluate] No ground truth loaded, skipping." << std::endl;
        return;
    }

    if (frame_scores_.empty()) {
        std::cout << "[Evaluate] No frames evaluated." << std::endl;
        return;
    }

    // 精度得分: 所有帧平均 (0-100)
    float total = 0.0f;
    for (float s : frame_scores_) {
        total += s;
    }
    float accuracy_score = total / static_cast<float>(frame_scores_.size());

    // 速度得分: FPS 映射到 0-100
    // 30 FPS 及以上为满分，线性衰减到 0 FPS = 0 分
    float fps_score = std::min(avg_fps / 30.0f, 1.0f) * 100.0f;

    // 综合得分: 精度 70% + 速度 30%
    float final_score = accuracy_score * 0.7f + fps_score * 0.3f;

    std::cout << "\n====== Evaluate ======" << std::endl;
    std::cout << "Sigma:           " << sigma_ << " px" << std::endl;
    std::cout << "Evaluated frames:" << frame_scores_.size() << std::endl;
    std::cout << "Accuracy score:  " << std::fixed << std::setprecision(2)
              << accuracy_score << " / 100  (weight 70%)" << std::endl;
    std::cout << "FPS score:       " << std::fixed << std::setprecision(2)
              << fps_score << " / 100  (weight 30%)" << std::endl;
    std::cout << "------" << std::endl;
    std::cout << "Final score:     " << std::fixed << std::setprecision(2)
              << final_score << " / 100" << std::endl;
    std::cout << "======================" << std::endl;
}
