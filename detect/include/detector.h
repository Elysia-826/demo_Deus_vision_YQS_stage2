#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>   // 改为onnxruntime的 C++ API 头文件

/// 单个装甲板检测结果
struct ArmorObject {
    cv::Rect2f bbox;
    int class_id;
    float confidence;
    cv::Point2f center;
};

/**
 * @brief 装甲板检测器
 *
 * 目标: 加载 ONNX 模型，对输入图像完成推理，输出每个检测目标的中心点
 */
class ArmorDetector {
public:
    ArmorDetector() = default;
    ~ArmorDetector() = default;

    bool init(const std::string& model_path,
              float conf_threshold = 0.5f,
              float nms_threshold = 0.45f,
              const cv::Size& input_size = cv::Size(640, 640));

    bool detect(const cv::Mat& frame, std::vector<ArmorObject>& results);

private:
    cv::Mat preprocess(const cv::Mat& frame);
    void postprocess(const cv::Mat& output, const cv::Size& frame_size, std::vector<ArmorObject>& results);
    
    // 新增 ONNX Runtime 成员
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    struct LetterBoxParams {
        float scale = 1.0f;
        float dx = 0;
        float dy = 0;
        cv::Size original_size;
    } letterbox_params_;

    float conf_threshold_ = 0.5f;
    float nms_threshold_ = 0.45f;
    cv::Size input_size_ = {640, 640};
    std::vector<std::string> class_names_;
};

