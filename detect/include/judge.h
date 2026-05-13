#pragma once

#include "target_selector.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>

/**
 * @brief Judge 日志模块 (热插拔)
 *
 * 将每帧输出的装甲板中心点写入 assets/log/ 下的 CSV 文件
 * 每行格式: frame_id, count, x1, y1, x2, y2, x3, y3, x4, y4
 * 不足 4 个的位置填 0
 *
 * 启用方式: cmake .. -DENABLE_JUDGE=ON
 */
class Judge {
public:
    Judge() = default;
    ~Judge();

    bool init(const std::string& log_dir = "assets/log");
    void log(int frame_id, const FrameResult& result);
    void close();

private:
    std::ofstream log_file_;
    bool opened_ = false;
};
