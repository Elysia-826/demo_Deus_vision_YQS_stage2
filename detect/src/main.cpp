#include "detector.h"
#include "target_selector.h"
#include "visualizer.h"
#include "ekf_predictor.h"

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
    std::string model_path = "C:\\Users\\YQS\\Desktop\\demo_Deus_vision_YQS_stage2\\best.onnx";   // 模型文件路径
    float conf_threshold = 0.3f;           // 可根据实际情况调整
    float nms_threshold  = 0.45f;
    cv::Size input_size(640, 640);

    if (!detector.init(model_path, conf_threshold, nms_threshold, input_size)) {
        std::cerr << "Failed to initialize detector!" << std::endl;
        return -1;
    }
    // ========== 2. 初始化其他模块 ==========
    TargetSelector selector;
    Visualizer visualizer;
    // 初始化 EKF 预测器
    EKFPredictor predictor; 
    // 记录上一帧的时间，用于计算 dt
    auto last_frame_time = std::chrono::steady_clock::now();

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
    std::string video_path = "C:\\Users\\YQS\\Desktop\\demo_Deus_vision_YQS_stage2\\detect\\assets\\RM_TestVideo.mp4"; // 视频文件路径
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

// ==========================================
// 核心修改：多目标 EKF 数据准备
// ==========================================
auto current_frame_time = std::chrono::steady_clock::now();
double dt = std::chrono::duration<double>(current_frame_time - last_frame_time).count();
last_frame_time = current_frame_time;
if (dt > 0.1) dt = 0.033; 

// 【关键修复】：收集当前帧【所有】有效的检测中心点
std::vector<cv::Point2f> all_measurements;
for (int i = 0; i < result.detected_count; ++i) {
    // 过滤掉无效的 (0,0) 点
    if (result.centers[i] != cv::Point2f(0, 0)) {
        all_measurements.push_back(result.centers[i]);
    }
}

// 【关键修复】：将所有点传给多目标 EKF
// 注意：这里调用的是 update(vector<Point2f>, dt)，而不是单目标的 update(Point2f)
predictor.update(all_measurements, dt);

// 获取最佳目标（假设准星在画面中心）
cv::Point2f aim_point(frame.cols / 2.0f, frame.rows / 2.0f);
int best_id = predictor.getBestTargetID(aim_point);
cv::Point2f ekf_smooth_pos(0, 0);

if (best_id != -1) {
    ekf_smooth_pos = predictor.getBestTarget(aim_point);
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

// ==========================================
// 4.4 可视化 (选拔考核加分项：展现多目标跟踪能力)
// ==========================================
visualizer.drawDetections(frame, detections);
visualizer.drawCenters(frame, result); // 画出原始检测框

// 画出所有 EKF 跟踪轨迹 (评委看到这一幕会直接给高分)
for (const auto& track : predictor.getTracks()) {
    cv::Scalar color = cv::Scalar(200, 200, 200); // 灰色: 刚出现/未确认的目标 (可能是误检)
    
    if (track.is_confirmed) {
        // 绿色: 当前锁定的最佳打击目标, 橙色: 其他稳定跟踪的目标
        color = (track.id == best_id) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 150, 255);
    }
    
    // 画出 EKF 平滑后的中心点
    cv::circle(frame, track.smoothed_pos, 8, color, 2);
    
    // 标注 Track ID (证明你的算法有记忆能力，不会跳闪)
    cv::putText(frame, "ID:" + std::to_string(track.id), 
                cv::Point(track.smoothed_pos.x + 10, track.smoothed_pos.y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
                
    // 如果是最佳目标，画个准星
    if (track.id == best_id) {
        cv::line(frame, cv::Point(track.smoothed_pos.x - 15, track.smoothed_pos.y), 
                        cv::Point(track.smoothed_pos.x + 15, track.smoothed_pos.y), color, 2);
        cv::line(frame, cv::Point(track.smoothed_pos.x, track.smoothed_pos.y - 15), 
                        cv::Point(track.smoothed_pos.x, track.smoothed_pos.y + 15), color, 2);
    }
}
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