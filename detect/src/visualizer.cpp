#include "visualizer.h"

void Visualizer::drawDetections(cv::Mat& frame,
                                const std::vector<ArmorObject>& detections) {
    for (size_t i = 0; i < detections.size(); i++) {
        const auto& obj = detections[i];
        // 绿色矩形框
        cv::rectangle(frame, obj.bbox, cv::Scalar(0, 255, 0), 2);
        // 置信度文本
        std::string label = std::to_string(int(obj.confidence * 100)) + "%";
        cv::putText(frame, label,
                    cv::Point(obj.bbox.x, obj.bbox.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }

    /*
    (void)frame;
    (void)detections;*/
}

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

    /*
    (void)frame;
    (void)fps;
    (void)detected_count;*/
}
