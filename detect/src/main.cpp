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
    // ========== 1. 初始化检测器 ==========
    ArmorDetector detector;
    std::string model_path = "C:\\Users\\YQS\\OneDrive\\文档\\GitHub\\demo_Deus_vision_YQS_stage2\\detect\\best.onnx";   // 模型文件路径（放在 exe 同目录）
    float conf_threshold = 0.5f;            // 可根据实际情况调整
    float nms_threshold  = 0.45f;
    cv::Size input_size(640, 640);

    if (!detector.init(model_path, conf_threshold, nms_threshold, input_size)) {
        std::cerr << "Failed to initialize detector!" << std::endl;
        return -1;
    }
    // ========== 2. 初始化其他模块 ==========
    TargetSelector selector;
    Visualizer visualizer;

#ifdef ENABLE_JUDGE
    Judge judge;
    judge.init("assets/log");// 日志输出目录
#endif

#ifdef ENABLE_EVALUATE
    Evaluator evaluator;
    evaluator.init("assets/std.csv");// 标准答案文件
#endif

    /*ArmorDetector detector;
    TargetSelector selector;
    [[maybe_unused]] Visualizer vis;*/

    // ========== 3. 打开输入源（视频或摄像头） ==========
    cv::VideoCapture cap;
    std::string video_path = "assets/test_video.mp4";
    cap.open(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video, trying camera 0..." << std::endl;
        cap.open(0);
        if (!cap.isOpened()) {
            std::cerr << "Failed to open camera!" << std::endl;
            return -1;
        }
    }

    // ========== 4. 主循环 ==========
    cv::Mat frame;
    int total_frames = 0;//帧计数器,也用于fps计算
    int detected_frames = 0;//检测到装甲板的帧数
    float total_time_ms = 0.0f;//累计推理耗时(毫秒)
    int frame_id = 0;//当前帧的绝对序号,从1开始递增
    auto start_time = std::chrono::steady_clock::now();

    while (cap.read(frame)) {
        if (frame.empty()) break;
        total_frames++;
        frame_id++;// frame_id 和 total_frames 同步增长
        auto t0 = std::chrono::high_resolution_clock::now();// 单帧推理计时开始


        // 4.1 执行检测，获取结果列表
        std::vector<ArmorObject> detections;
        detector.detect(frame, detections);


        // 4.2 提取中心点
        FrameResult result = selector.update(detections);

        auto t1 = std::chrono::high_resolution_clock::now();
        total_time_ms += std::chrono::duration<float, std::milli>(t1 - t0).count();

        if (result.detected_count > 0) {
            detected_frames++;
        }

        // 4.3 评估 & 日志
// 这里必须传入 frame_id（视频绝对帧序号），而不是 total_frames（处理帧计数）。
// 原因：
//   - std.csv 中的帧号是绝对帧 ID，evaluator 需要精确匹配，传入 total_frames 会导致错位。
//   - frame_id 是帧的“标识”，total_frames 是处理的“统计”，两者职责不同。
//   - 如果将来跳帧或从中间开始处理，frame_id 依然能与标准答案对齐，total_frames 则不能。
#ifdef ENABLE_JUDGE
        judge.log(frame_id, result);
#endif

#ifdef ENABLE_EVALUATE
        evaluator.submit(frame_id, result);
#endif

         // 4.4 可视化
        visualizer.drawDetections(frame, detections);
        visualizer.drawCenters(frame, result);

        // 4.5 计算 FPS
        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now - start_time).count();
        float fps = elapsed > 0 ? total_frames / elapsed : 0;
        visualizer.drawHUD(frame, fps, result.detected_count);

        // 4.6 显示
        cv::imshow("Armor Detection", frame);
        if (cv::waitKey(1) == 27) break;   // ESC 退出
    }


    cap.release();
    //cv::destroyAllWindows();

/*#ifdef ENABLE_JUDGE
    judge.close();
#endif*/

    float detection_rate = (total_frames > 0)
        ? static_cast<float>(detected_frames) / static_cast<float>(total_frames)
        : 0.0f;
    float avg_fps = (total_time_ms > 0)
        ? (1000.0f * total_frames / total_time_ms)
        : 0.0f;

         // ========== 5. 输出最终结果 ==========
    auto end_time = std::chrono::steady_clock::now();
    float total_elapsed = std::chrono::duration<float>(end_time - start_time).count();
    float avg_fps = total_elapsed > 0 ? total_frames / total_elapsed : 0;
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
#ifdef ENABLE_JUDGE
    judge.close();
#endif

    cv::destroyAllWindows();
    return 0;
}
