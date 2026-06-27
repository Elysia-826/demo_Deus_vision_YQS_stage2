#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
/**
 * @brief EKF 预测模块 (大二扩展)
 *
 * 目标: 对装甲板目标的平移与旋转进行建模和短时预测
 *
 * 你需要自行设计:
 *   - 状态向量
 *   - 状态转移模型
 *   - 观测模型
 *   - 噪声参数
 *
 * 并将此模块接入主程序的处理流程中
 */
class EKFPredictor {
    // 自行设计与实现
    public:
    // 模型枚举：CA(平移/匀加速), CTRV(旋转/小陀螺)
    enum ModelType { CA, CTRV };

    EKFPredictor();
    
    // 初始化（使用第一帧观测位置）
    void init(const cv::Point2f& initial_pos);
    
    // 预测 & 更新
    cv::Point2f predict(double dt);
    cv::Point2f update(const cv::Point2f& measurement);
    
    // 热切换模型
    void switchTo(ModelType target_model);
    ModelType getCurrentModel() const { return current_model_; }

    // 检查是否已初始化
    bool isInitialized() const { return is_initialized_; }
    
    // 状态获取
    cv::Point2f getPosition() const;
    cv::Point2f getVelocity() const; // CA返回(vx,vy), CTRV返回(v*cos(yaw), v*sin(yaw))
    double getAngularRate() const;   // 仅CTRV有效(小陀螺角速度), CA返回0

    // 自动切换建议（基于运动特征）
    bool shouldSwitchToCTRV() const;
    bool shouldSwitchToCA() const;

private:
    ModelType current_model_;
    bool is_initialized_;

    // CA 模型状态 (6维): [x, y, vx, vy, ax, ay]
    cv::Mat x_ca_, P_ca_, F_ca_, H_ca_, Q_ca_, R_;
    
    // CTRV 模型状态 (5维): [x, y, v, yaw, omega]
    cv::Mat x_ctrv_, P_ctrv_, H_ctrv_, Q_ctrv_;
    
    // 内部预测/更新实现
    void predictCA(double dt);
    void predictCTRV(double dt);
    void updateCommon(const cv::Mat& z, cv::Mat& x, cv::Mat& P, const cv::Mat& H, const cv::Mat& R);
    
    // 状态转换与协方差安全重置
    void caToCtrv();
    void ctrvToCa();
    
    // 工具函数
    static double wrapAngle(double angle);
};
