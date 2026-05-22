#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
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
    ArmorDetector();
    ~ArmorDetector() = default;

    bool init(const std::string& model_path,
              float conf_threshold = 0.5f,
              float nms_threshold = 0.45f,
              const cv::Size& input_size = cv::Size(640, 640));

    bool detect(const cv::Mat& frame, std::vector<ArmorObject>& results);

private:
    // 将图像转为 CHW 展开的平铺 float 数组
    // 确保这里是 cv::Mat
    cv::Mat preprocess(const cv::Mat& frame);
    // 修改为接收 Ort::Value 或 raw 数据的后处理逻辑
    // 确保第一个参数写成 const cv::Mat& output
    void postprocess(const cv::Mat& output, const cv::Size& frame_size, std::vector<ArmorObject>& results);
    
    // ONNX Runtime 核心成员
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    // 模型输入/输出节点名与 Shape 缓存
    std::vector<std::string> input_node_names_alloc_;
    std::vector<std::string> output_node_names_alloc_;
    std::vector<const char*> input_node_names_;
    std::vector<const char*> output_node_names_;
    std::vector<int64_t> input_shape_;

    // 输入缓冲区内存复用（避免多线程或循环调用时频繁分配内存带来的耗时）
    std::vector<float> input_tensor_values_;

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