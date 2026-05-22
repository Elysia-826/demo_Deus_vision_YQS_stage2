#include "detector.h"
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <cmath>

// ============================================================
// 初始化：加载 ONNX 模型，启用 CUDA
// ============================================================
bool ArmorDetector::init(const std::string& model_path,
                         float conf_threshold,
                         float nms_threshold,
                         const cv::Size& input_size) {
    conf_threshold_ = conf_threshold;
    nms_threshold_  = nms_threshold;
    input_size_     = input_size;

    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "armor_detect");

        Ort::SessionOptions session_options;
        // 尝试添加 CUDA 提供者
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        session_options.AppendExecutionProvider_CUDA(cuda_options);
        std::cout << "Using CUDA inference" << std::endl;

        // 将模型路径转为宽字符串
        std::wstring wmodel(model_path.begin(), model_path.end());
        // 直接用 reset 构造，避免模板推导错误
        session_.reset(new Ort::Session(*env_, wmodel.c_str(), session_options));
        std::cout << "Model loaded successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize ONNX Runtime: " << e.what() << std::endl;
        return false;
    }

    class_names_ = {"blue1", "blue3", "bluesb", "red1", "red3", "redsb"};
    return true;
}

// ============================================================
// 预处理：保持不变（letterbox + blob）
// ============================================================
cv::Mat ArmorDetector::preprocess(const cv::Mat& frame) {
    float scale = std::min((float)input_size_.width / frame.cols,
                           (float)input_size_.height / frame.rows);
    int new_w = (int)(frame.cols * scale);
    int new_h = (int)(frame.rows * scale);

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_w, new_h));

    cv::Mat letterbox(input_size_, CV_8UC3, cv::Scalar(114, 114, 114));
    int dx = (input_size_.width - new_w) / 2;
    int dy = (input_size_.height - new_h) / 2;
    resized.copyTo(letterbox(cv::Rect(dx, dy, new_w, new_h)));

    letterbox_params_ = {scale, static_cast<float>(dx), static_cast<float>(dy), frame.size()};

    cv::Mat blob = cv::dnn::blobFromImage(letterbox, 1.0/255.0, input_size_, cv::Scalar(), true, false);
    return blob;
}

// ============================================================
// 检测：用 ONNX Runtime 推理
// ============================================================
bool ArmorDetector::detect(const cv::Mat& frame, std::vector<ArmorObject>& results) {
    results.clear();
    if (frame.empty() || !session_) return false;

    cv::Mat blob = preprocess(frame);

    // 准备输入张量
    std::vector<int64_t> input_shape = {1, 3, input_size_.height, input_size_.width};
    size_t input_data_size = blob.total() * sizeof(float);
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, (float*)blob.data, input_data_size, input_shape.data(), input_shape.size());

    // 推理
    const char* input_names[] = {"images"};
    const char* output_names[] = {"output0"};
    try {
        auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                            input_names, &input_tensor, 1,
                                            output_names, 1);
        auto& output = output_tensors[0];
        auto shape = output.GetTensorTypeAndShapeInfo().GetShape();
        // shape: [1, 10, 8400]
        int num_detections = static_cast<int>(shape[2]);
        int num_channels = static_cast<int>(shape[1]);

        // 将输出数据复制到 cv::Mat（按行存储，每行是一个通道）
        cv::Mat output_mat(num_detections, num_channels, CV_32FC1, output.GetTensorMutableData<float>());
        cv::Mat output_transposed = output_mat.t();  // 转置为 [channels, detections]
        postprocess(output_transposed, frame.size(), results);
    } catch (const std::exception& e) {
        std::cerr << "Inference failed: " << e.what() << std::endl;
        return false;
    }

    return true;
}

// ============================================================
// 后处理：通道5~9取最大置信度，坐标还原，NMS
// ============================================================
void ArmorDetector::postprocess(const cv::Mat& output,
                                const cv::Size& frame_size,
                                std::vector<ArmorObject>& results) {
    // 此时 output 是 [10, 8400] 的矩阵（10行，8400列）
    const int num_channels = output.rows;
    const int num_detections = output.cols;
    float* data = (float*)output.data;

    float* ch_cx   = data + 0 * num_detections;
    float* ch_cy   = data + 1 * num_detections;
    float* ch_w    = data + 2 * num_detections;
    float* ch_h    = data + 3 * num_detections;
    float* ch_cls0 = data + 5 * num_detections;
    float* ch_cls1 = data + 6 * num_detections;
    float* ch_cls2 = data + 7 * num_detections;
    float* ch_cls3 = data + 8 * num_detections;
    float* ch_cls4 = data + 9 * num_detections;

    float scale = letterbox_params_.scale;
    float dx = letterbox_params_.dx;
    float dy = letterbox_params_.dy;
    float img_w = (float)frame_size.width;
    float img_h = (float)frame_size.height;

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;

    for (int i = 0; i < num_detections; i++) {
        float cx = ch_cx[i], cy = ch_cy[i], w = ch_w[i], h = ch_h[i];
        float obj_conf = std::max({ch_cls0[i], ch_cls1[i], ch_cls2[i], ch_cls3[i], ch_cls4[i]});

        if (obj_conf < conf_threshold_) continue;

        float x1 = (cx - w / 2.0f - dx) / scale;
        float y1 = (cy - h / 2.0f - dy) / scale;
        float bw = w / scale;
        float bh = h / scale;

        x1 = std::max(0.0f, x1);
        y1 = std::max(0.0f, y1);
        bw = std::min(bw, img_w - x1);
        bh = std::min(bh, img_h - y1);
        if (bw <= 0 || bh <= 0) continue;

        int ix = (int)std::max(0.0f, x1);
        int iy = (int)std::max(0.0f, y1);
        int iw = (int)std::max(0.0f, bw);
        int ih = (int)std::max(0.0f, bh);

        boxes.emplace_back(ix, iy, iw, ih);
        confidences.push_back(obj_conf);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, nms_threshold_, indices);

    for (int idx : indices) {
        ArmorObject obj;
        obj.bbox       = cv::Rect2f(boxes[idx]);
        obj.confidence = confidences[idx];
        obj.class_id   = 0;
        obj.center     = cv::Point2f(boxes[idx].x + boxes[idx].width / 2.0f,
                                      boxes[idx].y + boxes[idx].height / 2.0f);
        results.push_back(obj);
    }
}