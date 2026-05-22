#include "detector.h"
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <cmath>

ArmorDetector::ArmorDetector() 
    : env_(ORT_LOGGING_LEVEL_WARNING, "armor_detect"), 
      session_(nullptr) {
}

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
        // 1. 配置 CUDA 加速选项
        Ort::SessionOptions session_options;
        
        try {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0; // 默认使用0号显卡
            // 优化比赛时的显存分配策略，避免突发掉帧
            cuda_options.arena_extend_strategy = 0; 
            
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "[SUCCESS] Using CUDA inference for RoboMaster Armor Detect." << std::endl;
        } catch (const std::exception& cuda_ex) {
            std::cerr << "[WARNING] CUDA 注册失败，自动回退到纯 CPU 模式! 原因: " << cuda_ex.what() << std::endl;
        }

        // 2. 基本图优化配置
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // 3. 将模型路径转为宽字符串（Windows 平台必须）
#ifdef _WIN32
        std::wstring wmodel(model_path.begin(), model_path.end());
        session_ = std::make_unique<Ort::Session>(env_, wmodel.c_str(), session_options);
#else
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);
#endif
        std::cout << "Model loaded successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize ONNX Runtime: " << e.what() << std::endl;
        return false;
    }

    // 装甲板类别：蓝方1、蓝方3、蓝方哨兵；红方1、红方3、红方哨兵
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

    // blobFromImage 内部会完成 HWC 到 CHW 的转置并进行归一化
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

    // 1. 准备输入张量与内存信息 (使用 OrtArenaAllocator 允许 ORT 内部管理 GPU 与 CPU 的高效映射)
    std::vector<int64_t> input_shape = {1, 3, input_size_.height, input_size_.width};
    size_t input_tensor_size = blob.total(); 
    
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, reinterpret_cast<float*>(blob.data), input_tensor_size, input_shape.data(), input_shape.size());

    // 2. 指定输入输出节点名称
    const char* input_names[] = {"images"};
    const char* output_names[] = {"output0"};

    try {
        // 3. 执行推理
        auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                            input_names, &input_tensor, 1,
                                            output_names, 1);
        
        auto& output = output_tensors[0];
        auto shape = output.GetTensorTypeAndShapeInfo().GetShape();
        
        // 假设导出的 YOLO 结构 shape: [1, 10, 8400] -> 10个通道(cx,cy,w,h,conf,cls0~cls4), 8400个网格锚点
        int num_channels = static_cast<int>(shape[1]);
        int num_detections = static_cast<int>(shape[2]);

        // 将数据包在 cv::Mat 中进行高效矩阵转置
        cv::Mat output_mat(num_channels, num_detections, CV_32FC1, output.GetTensorMutableData<float>());
        cv::Mat output_transposed = output_mat.t();  // 转置后变成 [8400, 10]
        
        postprocess(output_transposed, frame.size(), results);
        
    } catch (const std::exception& e) {
        std::cerr << "ONNX Runtime inference failed: " << e.what() << std::endl;
        return false;
    }

    return true;
}

// ============================================================
// 后处理：提取最大置信度与其分类，坐标还原，NMS
// ============================================================
void ArmorDetector::postprocess(const cv::Mat& output,
                                const cv::Size& frame_size,
                                std::vector<ArmorObject>& results) {
    // 此时入参的 output 经过了转置，是 [8400, 10] 的矩阵（每行代表一个检测框锚点）
    const int num_detections = output.rows;
    const int num_channels = output.cols; 

    float scale = letterbox_params_.scale;
    float dx = letterbox_params_.dx;
    float dy = letterbox_params_.dy;
    float img_w = (float)frame_size.width;
    float img_h = (float)frame_size.height;

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    for (int i = 0; i < num_detections; i++) {
        // 获取当前行的首指针
        const float* row_ptr = output.ptr<float>(i);
        
        // 提取类别概率数组 (通道 5 往后是各个类别的得分)
        // 针对你的通道结构 5~9 个通道：
        const float* class_scores_ptr = row_ptr + 5;
        
        // 寻找 5 个装甲板类别里的最大得分值和对应的本地 ID
        auto max_score_iter = std::max_element(class_scores_ptr, row_ptr + num_channels);
        float obj_conf = *max_score_iter;

        // 阈值过滤
        if (obj_conf < conf_threshold_) continue;

        // 计算正确的 class_id 索引
        int label_id = std::distance(class_scores_ptr, max_score_iter);

        // 提取基础 BBox 信息
        float cx = row_ptr[0];
        float cy = row_ptr[1];
        float w  = row_ptr[2];
        float h  = row_ptr[3];

        // 还原回原图坐标 (去除 letterbox 的黑边和缩放比例)
        float x1 = (cx - w / 2.0f - dx) / scale;
        float y1 = (cy - h / 2.0f - dy) / scale;
        float bw = w / scale;
        float bh = h / scale;

        // 边界安全裁剪，防止画框越界崩溃
        x1 = std::max(0.0f, x1);
        y1 = std::max(0.0f, y1);
        bw = std::min(bw, img_w - x1);
        bh = std::min(bh, img_h - y1);
        if (bw <= 0 || bh <= 0) continue;

        boxes.emplace_back(static_cast<int>(x1), static_cast<int>(y1), static_cast<int>(bw), static_cast<int>(bh));
        confidences.push_back(obj_conf);
        class_ids.push_back(label_id);
    }

    // 4. 执行 OpenCV 的非极大值抑制（NMS）
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, nms_threshold_, indices);

    // 5. 组合最终检测输出
    for (int idx : indices) {
        ArmorObject obj;
        obj.bbox       = cv::Rect2f(boxes[idx]);
        obj.confidence = confidences[idx];
        obj.class_id   = class_ids[idx]; // 成功修复！现在能分清红蓝和数字了
        obj.center     = cv::Point2f(boxes[idx].x + boxes[idx].width / 2.0f,
                                     boxes[idx].y + boxes[idx].height / 2.0f);
        results.push_back(obj);
    }
}