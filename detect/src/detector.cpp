#include "detector.h"
#include <opencv2/dnn.hpp>    // 提供 cv::dnn::Net, blobFromImage, NMSBoxes 等
#include <algorithm>          // std::min, std::max
#include <iostream>           // std::cerr, std::cout

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
// 后处理：解析模型输出 → 坐标还原 → NMS → 填充 ArmorObject
// ============================================================
void ArmorDetector::postprocess(const cv::Mat& output,
                                const cv::Size& frame_size,
                                std::vector<ArmorObject>& results) {
     // ★ 打印输出张量形状
    std::cout << "Output dims: " << output.size[0] << " " << output.size[1] << " " << output.size[2] << std::endl;

    const int num_detections = output.size[2];   // 8400
    const int num_channels  = output.size[1];    // 这里会打印出来，看是否等于6
    float* data = (float*)output.data;

    // ★ 如果通道数不是6，打印实际通道数
    int actual_channels = output.size[1];
    if (actual_channels != 6) {
        std::cout << "Actual channels: " << actual_channels << std::endl;
    }

    // ★ 打印前2个检测点的前10个通道值
    for (int i = 0; i < 2 && i < num_detections; i++) {
        float* row = data + i * actual_channels;
        std::cout << "Detection " << i << ": ";
        for (int j = 0; j < std::min(10, actual_channels); j++) {
            std::cout << row[j] << " ";
        }
        std::cout << std::endl;
    }
    // 输出张量形状: [1, 6, 8400]
    //   8400 = 80×80 + 40×40 + 20×20  (三个检测头 anchor 总数)
    //   6 个通道: cx, cy, w, h, obj_conf, 以及可能的类别置信度
    const int num_detections = output.size[2];   // 8400
    const int num_channels  = output.size[1];    // 6
    float* data = (float*)output.data;           // 获取原始数据指针

    // 从成员变量中取出预处理时保存的 letterbox 参数
    float scale  = letterbox_params_.scale;
    float dx     = letterbox_params_.dx;
    float dy     = letterbox_params_.dy;
    float img_w  = (float)frame_size.width;
    float img_h  = (float)frame_size.height;

    // 使用整数矩形以匹配 cv::dnn::NMSBoxes 的可用重载
    std::vector<cv::Rect> boxes;      // 存放还原后的框坐标（整数）
    std::vector<float> confidences;     // 存放每个框的置信度

    // 遍历所有 8400 个候选框
    for (int i = 0; i < num_detections; i++) {
        float* row = data + i * num_channels;   // 指向第 i 个检测点的数据

        // 从输出中读取：中心点 xy、宽高 wh、目标性得分 obj_conf
        float cx = row[0];
        float cy = row[1];
        float w  = row[2];
        float h  = row[3];
        float obj_conf = row[4];       // 物体存在的置信度

        // 置信度过滤：低于阈值的直接丢弃
        // 置信度过滤：先做一次强力过滤，排除明显无效的检测
        if (obj_conf < 0.3f) continue;   // 0.3 比默认的 0.5 更低，但能过滤掉大量噪声

        // ---------- 坐标还原 ----------
        // 模型输出的是在 640×640 letterbox 画布上的坐标。
        // 需要还原到原始图像尺寸。
        // 还原公式：x_original = (x_letterbox - dx) / scale
        float x1 = (cx - w / 2.0f - dx) / scale;
        float y1 = (cy - h / 2.0f - dy) / scale;
        float bw = w / scale;
        float bh = h / scale;

        // 边界裁剪：确保框不超出图像范围
        x1 = std::max(0.0f, x1);
        y1 = std::max(0.0f, y1);
        bw = std::min(bw, img_w - x1);
        bh = std::min(bh, img_h - y1);

        // 过滤掉宽或高为 0 的无效框
        if (bw <= 0 || bh <= 0) continue;

        // 将浮点坐标转换为整数矩形（向下取整左上，限制宽高为非负整数）
        int ix = (int)std::max(0.0f, x1);
        int iy = (int)std::max(0.0f, y1);
        int iw = (int)std::max(0.0f, bw);
        int ih = (int)std::max(0.0f, bh);
        boxes.emplace_back(ix, iy, iw, ih);
        confidences.push_back(obj_conf);
    }

    // ---------- NMS (非极大值抑制) ----------
    // 同一个物体可能被多个 anchor 框检测到，NMS 保留最佳框，去除重叠框
    std::vector<int> indices;       // 保存被保留框的索引
    // 使用匹配的重载，指定 eta 和 top_k（默认 1.0f, 0）
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, nms_threshold_, indices, 1.0f, 0);

    // ---------- 填充结果 ----------
    for (int idx : indices) {
        ArmorObject obj;
        obj.bbox       = boxes[idx];               // 矩形框
        obj.confidence = confidences[idx];          // 置信度
        obj.class_id   = 0;                         // 类别 ID（暂未使用，可后续改进）
        // 计算中心点：框左上角 + 宽高的一半
        obj.center = cv::Point2f(
            obj.bbox.x + obj.bbox.width  / 2.0f,
            obj.bbox.y + obj.bbox.height / 2.0f
        );
        results.push_back(obj);
        std::cout << "Detected " << results.size() << " objects" << std::endl;
    }

    // NMS 之后
    std::cout << "After NMS: " << indices.size() << " objects" << std::endl;
    /*(void)output;
    (void)frame_size;
    (void)results;*/
}
