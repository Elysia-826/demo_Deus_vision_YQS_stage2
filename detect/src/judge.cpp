#include "judge.h"
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <sstream>

Judge::~Judge() {
    close();
}

bool Judge::init(const std::string& log_dir) {
    namespace fs = std::filesystem;

    if (!fs::exists(log_dir)) {
        fs::create_directories(log_dir);
    }

    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << log_dir << "/judge_" << std::put_time(std::localtime(&t), "%Y%m%d_%H%M%S") << ".csv";

    log_file_.open(ss.str());
    if (!log_file_.is_open()) {
        std::cerr << "[Judge] 打不开  " << ss.str() << std::endl;
        return false;
    }

    log_file_ << "frame_id,count,x1,y1,x2,y2,x3,y3,x4,y4" << std::endl;
    opened_ = true;
    std::cout << "[Judge] 已日志 " << ss.str() << std::endl;
    return true;
}

void Judge::log(int frame_id, const FrameResult& result) {
    if (!opened_) return;

    log_file_ << frame_id << "," << result.detected_count;
    for (int i = 0; i < MAX_ARMOR_COUNT; i++) {
        log_file_ << "," << std::fixed << std::setprecision(2)
                  << result.centers[i].x << "," << result.centers[i].y;
    }
    log_file_ << std::endl;
}

void Judge::close() {
    if (opened_) {
        log_file_.close();
        opened_ = false;
    }
}
