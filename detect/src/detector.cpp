#include "detector.h"
#include <opencv2/dnn.hpp>    // 提供 cv::dnn::Net, blobFromImage, NMSBoxes 等
#include <algorithm>          // std::min, std::max
#include <iostream>           // std::cerr, std::cout
#include <cmath>              // std::sqrt, std::pow

bool ArmorDetector::init(const std::string& model_path,
                         float conf_threshold,
                         float nms_threshold,
                         const cv::Size& input_size) {
    //(void)model_path;
    // 1. 将用户传入的参数保存到成员变量，后续推理时使用
    conf_threshold_ = conf_threshold; // 置信度阈值：低于此值的检测框会被丢弃
    nms_threshold_ = nms_threshold;// NMS 的 IoU 阈值：重叠高于此值的框将被合并
    input_size_ = input_size;// 模型要求的输入尺寸，例如 640x640

    // 2. 使用 OpenCV DNN 模块加载 ONNX 模型文件
    //    readNetFromONNX 会返回一个 cv::dnn::Net 对象，失败时内部为空
    net_ = cv::dnn::readNetFromONNX(model_path);
    // 尝试 OpenCL 后端（对多数 GPU 都有效，包含 NVIDIA）
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
    if (net_.empty()) {
        std::cerr << "Failed to load ONNX model: " << model_path << std::endl;
        return false;   // 加载失败，返回 false 通知调用者
    }
// 3. 初始化类别名称列表（顺序必须与你训练时的 data.yaml 中 names 列表一致）
    class_names_ = {"blue1", "blue3", "bluesb", "red1", "red3", "redsb"};

    std::cout << "Model loaded successfully!" << std::endl;
    return true;   // 初始化成功
}

// ============================================================
// 预处理：将任意尺寸的 BGR 图像转换为模型需要的 blob
// ============================================================
cv::Mat ArmorDetector::preprocess(const cv::Mat& frame) {
    // --- 第一步：Letterbox 缩放 ---
    // 计算缩放比例，选择较小的那个方向，保证图像完整显示在 640x640 内，不变形
    float scale = std::min(
        (float)input_size_.width  / frame.cols,   // 宽度方向能放大的倍数
        (float)input_size_.height / frame.rows    // 高度方向能放大的倍数
    );

    int new_w = (int)(frame.cols * scale);
    int new_h = (int)(frame.rows * scale);

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_w, new_h));

    // 创建一张 640x640 的灰色画布（114,114,114 是 YOLO 训练时常用的填充色）
    cv::Mat letterbox(input_size_, CV_8UC3, cv::Scalar(114, 114, 114));

    // 计算偏移量，将缩放后的图像放到画布正中央
    int dx = (input_size_.width  - new_w) / 2;
    int dy = (input_size_.height - new_h) / 2;
    resized.copyTo(letterbox(cv::Rect(dx, dy, new_w, new_h)));

    // 保存 letterbox 参数，供后处理时还原坐标
    letterbox_params_ = {scale, static_cast<float>(dx), static_cast<float>(dy), frame.size()};

    // --- 第二步：转换为 blob ---
    // blobFromImage 完成三件事：
    //   ① 将像素值从 [0, 255] 归一化到 [0, 1]
    //   ② BGR → RGB 转换 (第五个参数 true)
    //   ③ 将 HWC 排布变成 CHW 排布（通道优先）
    cv::Mat blob = cv::dnn::blobFromImage(
        letterbox,          // 输入图像
        1.0 / 255.0,        // 缩放因子
        input_size_,        // 输出尺寸（这里与输入相同）
        cv::Scalar(),       // 不减去均值
        true,               // swapRB: BGR to RGB
        false               // 不裁剪
    );

    return blob;

    //(void)frame;
    //return cv::Mat();
}


// ============================================================
// 检测主函数：输入一帧图像，输出检测到的装甲板列表
// ============================================================
bool ArmorDetector::detect(const cv::Mat& frame, std::vector<ArmorObject>& results) {
    results.clear();
    if (frame.empty()) return false;
    // 1. 前处理：将图像转为模型需要的格式
    cv::Mat blob = preprocess(frame);

    // 2. 将 blob 设置到网络的输入层，名字通常为 "images"
    net_.setInput(blob);

    // 3. 执行前向推理，获取所有输出层的输出
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());

    // 4. 后处理：解析输出张量，还原坐标，执行 NMS，填充结果
    postprocess(outputs[0], frame.size(), results);

    return true;
}

// ============================================================
// 后处理（兼容两种 ONNX 模型格式）
//
// 与原始框架的区别：
//   1. 增加了对“带内嵌 NMS 的模型”的支持（输出 [1, 7, 300]）
//   2. 原始模型路径下增加了无效框过滤和整数转换
//   3. 函数签名和输出格式与原始框架完全兼容
// ============================================================
void ArmorDetector::postprocess(const cv::Mat& output,
                                const cv::Size& frame_size,
                                std::vector<ArmorObject>& results) {
    const int num_detections = output.size[2]; //8400(原始)或300(带NMS)
    const int num_channels  = output.size[1]; //10(原始)或7(带NMS)
    float* data = (float*)output.data;

    // ============================================================
    // 路径 A：带内嵌 NMS 的模型（输出形状 [1, N, 300]，N 通常为 6 或 7）
    //   每行格式：[x1, y1, x2, y2, conf, cls] 或 [x1, y1, x2, y2, conf, cls_conf, cls]
    //   坐标已是原始图像坐标，无需 letterbox 还原，无需手动 NMS
    // ============================================================
    if (num_channels <= 7) {
        for (int i = 0; i < num_detections; i++) {
            float* row = data + i * num_channels;

            float x1 = row[0];
            float y1 = row[1];
            float x2 = row[2];
            float y2 = row[3];
            float obj_conf = row[4];

            // 置信度过滤
            if (obj_conf < conf_threshold_) continue;

            // 计算宽高
            float bw = x2 - x1;
            float bh = y2 - y1;

            // 边界裁剪
            x1 = std::max(0.0f, x1);
            y1 = std::max(0.0f, y1);
            bw = std::min(bw, (float)frame_size.width - x1);
            bh = std::min(bh, (float)frame_size.height - y1);

            // 无效框过滤
            if (bw <= 0 || bh <= 0) continue;

            ArmorObject obj;
            obj.bbox       = cv::Rect2f(x1, y1, bw, bh);
            obj.confidence = obj_conf;
            obj.class_id   = (num_channels >= 7) ? (int)row[6] : 0;
            obj.center     = cv::Point2f(x1 + bw / 2.0f, y1 + bh / 2.0f);
            results.push_back(obj);
        }
        return;
    }

    // ============================================================
    // 路径 B：原始 YOLOv11 模型（输出形状 [1, 10, 8400]）
    //   每行格式：[cx, cy, w, h, obj_conf, cls0, cls1, cls2, cls3, cls4]
    //   坐标是 640×640 letterbox 坐标，需还原 + 手动 NMS
    // ============================================================

    // 读取 letterbox 参数（preprocess 时保存）
    float scale  = letterbox_params_.scale;
    float dx     = letterbox_params_.dx;
    float dy     = letterbox_params_.dy;
    float img_w  = (float)frame_size.width;
    float img_h  = (float)frame_size.height;

// 通道映射（已验证）
    // 0:cx  1:cy  2:w  3:h  4:?  5:cls0  6:cls1  7:cls2  8:cls3  9:cls4
    float* ch_cx   = data + 0 * num_detections;
    float* ch_cy   = data + 1 * num_detections;
    float* ch_w    = data + 2 * num_detections;
    float* ch_h    = data + 3 * num_detections;
    float* ch_cls0 = data + 5 * num_detections;  // 类别0置信度
    float* ch_cls1 = data + 6 * num_detections;  // 类别1置信度
    float* ch_cls2 = data + 7 * num_detections;  // 类别2置信度
    float* ch_cls3 = data + 8 * num_detections;  // 类别3置信度
    float* ch_cls4 = data + 9 * num_detections;  // 类别4置信度

    std::vector<cv::Rect> boxes;      // 整数矩形框
    std::vector<float> confidences;   // 置信度

    for (int i = 0; i < num_detections; i++) {
        float* row = data + i * num_channels;

        // 读取 anchor 参数
        float cx = ch_cx[i];
        float cy = ch_cy[i];
        float w  = ch_w[i];
        float h  = ch_h[i];
         // 置信度取各类别得分中的最大值
        float obj_conf = std::max({ch_cls0[i], ch_cls1[i], ch_cls2[i],
                                   ch_cls3[i], ch_cls4[i]});

        // 用 sigmoid 将 logits 转换为 0~1 的概率
        float obj_conf_sigmoid = 1.0f / (1.0f + std::exp(-obj_conf));

        // 置信度过滤：用转换后的概率值进行判断
        if (obj_conf_sigmoid < conf_threshold_) continue;

        // 置信度过滤
        if (obj_conf < conf_threshold_) continue;

        // 坐标还原：从 letterbox 坐标 → 原始图像坐标
        // 公式：x_original = (x_letterbox - dx) / scale
        float x1 = (cx - w / 2.0f - dx) / scale;
        float y1 = (cy - h / 2.0f - dy) / scale;
        float bw = w / scale;
        float bh = h / scale;

        // 边界裁剪
        x1 = std::max(0.0f, x1);
        y1 = std::max(0.0f, y1);
        bw = std::min(bw, img_w - x1);
        bh = std::min(bh, img_h - y1);

        // 无效框过滤
        if (bw <= 0 || bh <= 0) continue;

        // 转为整数矩形（更符合像素坐标语义）
        int ix = (int)std::max(0.0f, x1);
        int iy = (int)std::max(0.0f, y1);
        int iw = (int)std::max(0.0f, bw);
        int ih = (int)std::max(0.0f, bh);

        boxes.emplace_back(ix, iy, iw, ih);
        confidences.push_back(obj_conf);
    }

    // 手动 NMS（非极大值抑制）
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, nms_threshold_, indices);

    // 填充结果
    for (int idx : indices) {
        ArmorObject obj;
        obj.bbox       = cv::Rect2f(boxes[idx]);
        obj.confidence = confidences[idx];
        obj.class_id   = 0;
        obj.center     = cv::Point2f(
            boxes[idx].x + boxes[idx].width  / 2.0f,
            boxes[idx].y + boxes[idx].height / 2.0f
        );
        results.push_back(obj);
    }

        //std::cout << "Detected " << results.size() << " objects" << std::endl;
    

    // NMS 之后
    //std::cout << "After NMS: " << indices.size() << " objects" << std::endl;
    /*(void)output;
    (void)frame_size;
    (void)results;*/
}