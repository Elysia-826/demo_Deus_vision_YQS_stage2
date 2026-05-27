#include <iostream>
#include "visualizer.h"

void Visualizer::drawDetections(cv::Mat& frame,
                                const std::vector<ArmorObject>& detections) {
    // 定义一个类别名称映射表
    // class_id能自动抓到对应的名字
    const std::vector<std::string> CLASS_NAMES = {
        "blue1", "blue3", "bluesb",
        "red1",  "red3",  "redsb"
    };
    for (size_t i = 0; i < detections.size(); i++) {
        const auto& obj = detections[i];
        // 绿色矩形框
        cv::rectangle(frame, obj.bbox, cv::Scalar(0, 255, 0), 2);
        // 置信度文本
        std::string label = std::to_string(int(obj.confidence * 100)) + "%";
        cv::putText(frame, label,
                    cv::Point(obj.bbox.x, obj.bbox.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

        // ========== 明确输出具体是哪辆车的中心点坐标 ==========
        // 自动计算当前目标的几何中心像素坐标
        // ========== 精准映射成字符串输出 ==========
        int center_x = int(obj.center.x);
        int center_y = int(obj.center.y);

        // 安全边界防御：防止 class_id 越界导致程序崩溃
        std::string target_name = "unknown";
        if (obj.class_id >= 0 && obj.class_id < static_cast<int>(CLASS_NAMES.size())) {
            target_name = CLASS_NAMES[obj.class_id];
        } else {
            target_name = "id_" + std::to_string(obj.class_id);
        }
        // 格式化输出到控制台，例如：[TARGET] red3 Center Coordinate: (412, 235)
       std::cout << "[TARGET] " << target_name
                  << " Center Coordinate: (" << center_x << ", " << center_y << ")" << std::endl;
        

    /*
    (void)frame;
    (void)detections;*/
}}

void Visualizer::drawCenters(cv::Mat& frame, const FrameResult& result) {
    for (int i = 0; i < result.detected_count; i++) {
        const auto& pt = result.centers[i];
        if (pt.x == 0 && pt.y == 0) continue;  // 无效点跳过
        // 红色十字准星，大小 20，线宽 2
        cv::drawMarker(frame, pt, cv::Scalar(0, 0, 255),
                       cv::MARKER_CROSS, 20, 2);
        // 在旁边标注序号
        cv::putText(frame, std::to_string(i),
                    pt + cv::Point2f(10, -10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    }   
    /*
    (void)frame;
    (void)result;*/
}

void Visualizer::drawHUD(cv::Mat& frame, float fps, int detected_count) {
    cv::putText(frame, "FPS: " + std::to_string(int(fps)),
                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 0), 2);
    cv::putText(frame, "Armors: " + std::to_string(detected_count),
                cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 0), 2);

    // ==========无目标时给出明确 LOST 状态==========
    if (detected_count == 0) {
        // 控制台提示
        std::cout << "[STATUS] NO RELIABLE TARGET DETECTED." << std::endl;
        
        // 画面左上角强力高亮绘制红色 LOST 状态，满足线下验收硬指标
        cv::putText(frame, "STATUS: LOST", cv::Point(10, 90), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    /*
    (void)frame;
    (void)fps;
    (void)detected_count;*/
}}