#include "detector.h"
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <memory>
#include <chrono>
#include <vector>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

static bool extractArmorCornersFast(const cv::Mat& roi, 
                                    std::vector<cv::Point2f>& corners, 
                                    cv::Point2f& center);

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

            // ====== 【核心注入：强迫 ORT 必须使用显卡硬件算子】 ======
            // 强制启用 cuDNN 卷积算法穷举搜索，激活 cuDNN 深度加速
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive; 
            // ========================================================
            
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "[SUCCESS] Using CUDA inference for RoboMaster Armor Detect." << std::endl;
        } catch (const std::exception& cuda_ex) {
            std::cerr << "[WARNING] CUDA 注册失败，自动回退到纯 CPU 模式! 原因: " << cuda_ex.what() << std::endl;
            return false; // 直接拦截，不给它用 CPU 苟活的机会
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

    // 使用高精度稳定纪元时钟，全量锁死时间节点，物理杜绝脏数据
    auto t_start = std::chrono::steady_clock::now();

    cv::Mat blob = preprocess(frame);

    // 捕捉预处理结束点
    auto t_pre_end = std::chrono::steady_clock::now();

    // 1. 准备输入张量与内存信息 
    //实车部署使用 OrtMemTypeDefault + CreateCpu 配合显式连续内存保护，确保显存访问安全且高效
    std::vector<int64_t> input_shape = {1, 3, input_size_.height, input_size_.width};
    size_t input_tensor_size = blob.total(); 
    
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, reinterpret_cast<float*>(blob.data), input_tensor_size, input_shape.data(), input_shape.size());

    // 2. 指定输入输出节点名称
    const char* input_names[] = {"images"};
    const char* output_names[] = {"output0"};

    auto t_infer_end = std::chrono::steady_clock::now(); // 初始化声明
    auto t_post_end  = std::chrono::steady_clock::now(); // 初始化声明

    try {
        // 3. 执行推理
        auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                            input_names, &input_tensor, 1,
                                            output_names, 1);
        t_infer_end = std::chrono::steady_clock::now(); //精准记录模型推理结束点
        
        // 4. 解析输出并进行后处理
        auto& output = output_tensors[0];
        auto shape = output.GetTensorTypeAndShapeInfo().GetShape();
        
        // 假设导出的 YOLO 结构 shape: [1, 10, 8400] -> 10个通道(cx,cy,w,h,conf,cls0~cls4), 8400个网格锚点
        int num_channels = static_cast<int>(shape[1]);
        int num_detections = static_cast<int>(shape[2]);

        // 将数据包在 cv::Mat 中进行高效矩阵转置
        cv::Mat output_mat(num_channels, num_detections, CV_32FC1, output.GetTensorMutableData<float>());
        cv::Mat output_transposed = output_mat.t();  // 转置后变成 [8400, 10]
        
        postprocess(output_transposed, frame, results);
        t_post_end = std::chrono::steady_clock::now(); //精准记录后处理结束点

    } catch (const std::exception& e) {
        std::cerr << "ONNX Runtime inference failed: " << e.what() << std::endl;
        return false;
    }
    // --- 计算并打印各环节详细耗时 (单位: 毫秒 ms) ---
    // 使用 std::chrono 标准耗时测算机制
    float t_pre   = std::chrono::duration<float, std::milli>(t_pre_end - t_start).count();
    float t_infer = std::chrono::duration<float, std::milli>(t_infer_end - t_pre_end).count();
    float t_post  = std::chrono::duration<float, std::milli>(t_post_end - t_infer_end).count();
    float t_total = std::chrono::duration<float, std::milli>(t_post_end - t_start).count();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "[Profile] Pre: " << t_pre << "ms | "
              << "Infer: " << t_infer << "ms | "
              << "Post: " << t_post << "ms | "
              << "Total: " << t_total << "ms" << std::endl;

    return true;
}

// ============================================================
// 后处理：提取最大置信度与其分类，坐标还原，NMS
// ============================================================
void ArmorDetector::postprocess(const cv::Mat& output,
                                const cv::Mat& frame,
                                std::vector<ArmorObject>& results) {
    // 此时入参的 output 经过了转置，是 [8400, 10] 的矩阵（每行代表一个检测框锚点）
    // 从 frame 中直接获取 size
    const cv::Size frame_size = frame.size(); 
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
        obj.class_id   = class_ids[idx]; 
        //(1)获取当前检测框
        cv::Rect r = boxes[idx];
        // 边界保护
        // YOLO 偶尔会输出超出图像边界的框，直接裁剪会导致 cv::Mat 崩溃。
        // 这里将框强制限制在原图尺寸内，车端防崩溃必备！
        r &= cv::Rect(0, 0, frame.cols, frame.rows);
        if (r.width <= 0 || r.height <= 0) continue; // 如果框被裁没了，直接跳过

        //(2)从原图中裁剪出检测框对应的 ROI 区域
        cv::Mat roi = frame(r);
       // (3) 调用新的车端特化提取函数
        std::vector<cv::Point2f> roi_corners;
        cv::Point2f roi_center;
        
        if (extractArmorCornersFast(roi, roi_corners, roi_center)) {
            // 【情况 A：传统 CV 提取成功,四角点完整,可用于PNP】
            // 将 ROI 局部坐标系下的点，转换回【原图全局坐标系】
            for (auto& pt : roi_corners) {
                pt.x += r.x;
                pt.y += r.y;
            }
            roi_center.x += r.x;
            roi_center.y += r.y;
            
            obj.corners = roi_corners; // 保存 4 个角点给 PnP
            obj.center = roi_center;   // 保存中心点给 EKF
             // 只有完美的、满足 PnP 要求的目标，才加入最终结果
            results.push_back(obj); 
        } else {
            // 【情况 B：传统 CV 提取失败 (如严重遮挡、反光过曝)】
            // 车端 EKF 友好兜底策略：
            // 1. 不要输出 (-1,-1) 或乱飘的噪点，直接把 YOLO 框的几何中心喂给 EKF。
            // 2. corners 留空。后续 PnP 模块判断 corners 为空时，跳过本帧解算，
            //    仅依靠 EKF 的预测方程 (Predict) 维持目标轨迹，防止滤波发散。
            //obj.center = cv::Point2f(r.x + r.width / 2.0f, r.y + r.height / 2.0f);
            // obj.corners 默认就是空的，不需要额外操作
        }

        results.push_back(obj);
    }
}

    // 在线矩计算 + 主方向投影提取四角点 (零动态分配, 确定延迟)
    // 返回 true 成功, false 失败
static bool extractArmorCornersFast(const cv::Mat& roi, 
                                    std::vector<cv::Point2f>& corners, 
                                    cv::Point2f& center) {
    using namespace cv;
    if (roi.empty()) return false;

    // 限制 ROI 最大尺寸，防止极端大框拖垮 CPU (比赛常用 120x80)
    const int MAX_W = 120, MAX_H = 80;
    Mat work_roi = roi;
    if (roi.cols > MAX_W || roi.rows > MAX_H) {
        cv::resize(roi, work_roi, Size(MAX_W, MAX_H), 0, 0, INTER_LINEAR);
    }

    // 1. 灰度 + Otsu 二值化
    Mat gray, binary;
    cvtColor(work_roi, gray, COLOR_BGR2GRAY);
    threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);

    // 反色保护：白像素超 40% 说明背景过曝，反转图像
    if (countNonZero(binary) > binary.total() * 0.4) {
        bitwise_not(binary, binary);
    }

    // 2. 在线计算一阶矩 & 二阶矩 (零分配, 单次遍历)
    double m00 = 0, m10 = 0, m01 = 0, m20 = 0, m02 = 0, m11 = 0;
    int valid_pixels = 0;

    for (int y = 0; y < binary.rows; ++y) {
        const uchar* row = binary.ptr<uchar>(y);
        for (int x = 0; x < binary.cols; ++x) {
            if (row[x] > 0) {
                m00 += 1;
                m10 += x;
                m01 += y;
                m20 += x * x;
                m02 += y * y;
                m11 += x * y;
                valid_pixels++;
            }
        }
    }

    // 像素过少直接失败 (避免除零和噪点干扰)
    if (valid_pixels < 40) return false;

    // 3. 计算质心 (中心点)
    double cx = m10 / m00;
    double cy = m01 / m00;
    center = Point2f(static_cast<float>(cx), static_cast<float>(cy));

    // 4. 计算协方差矩阵 & 主方向 (PCA 原理)
    double u20 = m20 / m00 - cx * cx;
    double u02 = m02 / m00 - cy * cy;
    double u11 = m11 / m00 - cx * cy;
    
    double angle = 0.5 * atan2(2 * u11, u20 - u02); // 装甲板主方向弧度
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    // 5. 沿主方向投影，找四个极值点 (替代脆弱的四象限法)
    float min_proj_x = 1e9f, max_proj_x = -1e9f;
    float min_proj_y = 1e9f, max_proj_y = -1e9f;
    
    // 投影坐标系下的极值点 (在原图坐标系中记录)
    Point2f pt_tl, pt_tr, pt_br, pt_bl;

    for (int y = 0; y < binary.rows; ++y) {
        const uchar* row = binary.ptr<uchar>(y);
        for (int x = 0; x < binary.cols; ++x) {
            if (row[x] > 0) {
                // 将像素点转换到以质心为原点、主方向为轴的局部坐标系
                float dx = x - cx;
                float dy = y - cy;
                float local_x =  dx * cos_a + dy * sin_a; // 沿灯条长度方向
                float local_y = -dx * sin_a + dy * cos_a; // 沿灯条宽度方向

                // 找局部坐标系下的四个极值
                if (local_x + local_y < min_proj_x + min_proj_y) { min_proj_x = local_x; min_proj_y = local_y; pt_tl = Point2f(x, y); }
                if (local_x - local_y > max_proj_x - min_proj_y) { max_proj_x = local_x; min_proj_y = local_y; pt_tr = Point2f(x, y); }
                if (local_x + local_y > max_proj_x + max_proj_y) { max_proj_x = local_x; max_proj_y = local_y; pt_br = Point2f(x, y); }
                if (local_x - local_y < min_proj_x - max_proj_y) { min_proj_x = local_x; max_proj_y = local_y; pt_bl = Point2f(x, y); }
            }
        }
    }

    // 6. 几何校验 (防止单灯条或严重遮挡误输出)
    float width  = std::hypotf(pt_tr.x - pt_tl.x, pt_tr.y - pt_tl.y);
    float height = std::hypotf(pt_bl.x - pt_tl.x, pt_bl.y - pt_tl.y);
    if (width < 15.0f || height < 8.0f || width / (height + 1e-5f) < 1.2f) {
        return false; // 不符合装甲板物理比例
    }

    // 7. 输出 PnP 严格顺序: 左上, 右上, 右下, 左下
    corners.clear();
    corners.reserve(4);
    corners.push_back(pt_tl);
    corners.push_back(pt_tr);
    corners.push_back(pt_br);
    corners.push_back(pt_bl);

    // 如果之前 resize 过，需要将坐标映射回原始 ROI 尺度
    if (work_roi.size() != roi.size()) {
        float sx = roi.cols / (float)work_roi.cols;
        float sy = roi.rows / (float)work_roi.rows;
        for (auto& pt : corners) { pt.x *= sx; pt.y *= sy; }
        center.x *= sx; center.y *= sy;
    }

    return true;
}