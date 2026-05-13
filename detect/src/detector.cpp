#include "detector.h"

bool ArmorDetector::init(const std::string& model_path,
                         float conf_threshold,
                         float nms_threshold,
                         const cv::Size& input_size) {
    (void)model_path;
    conf_threshold_ = conf_threshold;
    nms_threshold_ = nms_threshold;
    input_size_ = input_size;

    // TODO: 加载你的 ONNX 模型，初始化类别名称列表，设置推理后端



    return false;
}

bool ArmorDetector::detect(const cv::Mat& frame, std::vector<ArmorObject>& results) {
    results.clear();
    if (frame.empty()) return false;

    // TODO: 调用前处理 -> 送入网络推理 -> 调用后处理 -> 填充每个结果的 center 字段



    return false;
}

cv::Mat ArmorDetector::preprocess(const cv::Mat& frame) {
    // TODO: 将原图转换为模型所需的输入格式



    (void)frame;
    return cv::Mat();
}

void ArmorDetector::postprocess(const cv::Mat& output,
                                const cv::Size& frame_size,
                                std::vector<ArmorObject>& results) {
    // TODO: 解析网络输出张量，筛选置信度，还原坐标到原图尺寸，执行 NMS



    (void)output;
    (void)frame_size;
    (void)results;
}
