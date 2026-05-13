#include "detector.h"
#include "target_selector.h"
#include "visualizer.h"

#ifdef ENABLE_JUDGE
#include "judge.h"
#endif

#ifdef ENABLE_EVALUATE
#include "evaluate.h"
#endif

#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

#ifdef ENABLE_JUDGE
    Judge judge;
    judge.init("assets/log");
#endif

#ifdef ENABLE_EVALUATE
    Evaluator evaluator;
    evaluator.init("assets/std.csv");
#endif

    ArmorDetector detector;
    TargetSelector selector;
    [[maybe_unused]] Visualizer vis;

    // TODO: 初始化检测器 (加载模型)



    // TODO: 打开输入源 (图片/视频/摄像头)
    cv::VideoCapture cap;



    cv::Mat frame;
    int total_frames = 0;
    int detected_frames = 0;
    float total_time_ms = 0.0f;

    while (cap.read(frame)) {
        if (frame.empty()) break;
        total_frames++;

        auto t0 = std::chrono::high_resolution_clock::now();

        // TODO: 执行检测，获取结果列表
        std::vector<ArmorObject> detections;



        FrameResult result = selector.update(detections);

        auto t1 = std::chrono::high_resolution_clock::now();
        total_time_ms += std::chrono::duration<float, std::milli>(t1 - t0).count();

        if (result.detected_count > 0) {
            detected_frames++;
        }

#ifdef ENABLE_JUDGE
        judge.log(total_frames, result);
#endif

#ifdef ENABLE_EVALUATE
        evaluator.submit(total_frames, result);
#endif

        // TODO: 可视化绘制与显示



    }

    cap.release();
    cv::destroyAllWindows();

#ifdef ENABLE_JUDGE
    judge.close();
#endif

    float detection_rate = (total_frames > 0)
        ? static_cast<float>(detected_frames) / static_cast<float>(total_frames)
        : 0.0f;
    float avg_fps = (total_time_ms > 0)
        ? (1000.0f * total_frames / total_time_ms)
        : 0.0f;

    std::cout << "\n====== Result ======" << std::endl;
    std::cout << "Total frames:    " << total_frames << std::endl;
    std::cout << "Detected frames: " << detected_frames << std::endl;
    std::cout << "Detection rate:  " << std::fixed << std::setprecision(4)
              << detection_rate << " (" << detected_frames << "/" << total_frames << ")" << std::endl;
    std::cout << "Average FPS:     " << std::fixed << std::setprecision(2) << avg_fps << std::endl;
    std::cout << "====================" << std::endl;

#ifdef ENABLE_EVALUATE
    evaluator.printResult(avg_fps);
#endif

    return 0;
}
