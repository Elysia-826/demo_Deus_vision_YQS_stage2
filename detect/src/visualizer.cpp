#include "visualizer.h"

void Visualizer::drawDetections(cv::Mat& frame,
                                const std::vector<ArmorObject>& detections) {
    // TODO: 绘制检测结果



    (void)frame;
    (void)detections;
}

void Visualizer::drawCenters(cv::Mat& frame, const FrameResult& result) {
    // TODO: 绘制装甲板中心点



    (void)frame;
    (void)result;
}

void Visualizer::drawHUD(cv::Mat& frame, float fps, int detected_count) {
    // TODO: 绘制 FPS 和检测数量等状态信息



    (void)frame;
    (void)fps;
    (void)detected_count;
}
