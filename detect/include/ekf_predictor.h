#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>
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

    // 【内部结构体】：封装单个目标的 EKF 状态与数学模型
    struct Track {
        int id;
        ModelType current_model;
        bool is_initialized;
        
        // CA 模型状态 (6维)
        cv::Mat x_ca, P_ca, F_ca, H_ca, Q_ca, R;
        // CTRV 模型状态 (5维)
        cv::Mat x_ctrv, P_ctrv, H_ctrv, Q_ctrv;
        
        cv::Point2f smoothed_pos;
        int lost_frames;
        int hit_frames;
        bool is_confirmed; // 是否已确认为稳定目标（防止误检闪烁）
        
        // 单目标核心方法
        void initTrack(const cv::Point2f& initial_pos, const cv::Mat& R_mat, const cv::Mat& Q_ca_mat, const cv::Mat& Q_ctrv_mat);
        cv::Point2f predictTrack(double dt);
        cv::Point2f updateTrack(const cv::Point2f& measurement);
        cv::Point2f getPosition() const;
        
        // 内部数学运算
        void predictCA(double dt);
        void predictCTRV(double dt);
        void updateCommon(const cv::Mat& z, cv::Mat& x, cv::Mat& P, const cv::Mat& H, const cv::Mat& R_mat);
        void caToCtrv();
        void ctrvToCa();
        bool shouldSwitchToCTRV() const;
        bool shouldSwitchToCA() const;
    };

    EKFPredictor();
    
    // 【多目标核心接口】：传入当前帧所有检测到的中心点
    void update(const std::vector<cv::Point2f>& measurements, double dt);
    
    // 获取最佳打击目标 (距离 aim_point 最近的已确认目标)
    cv::Point2f getBestTarget(const cv::Point2f& aim_point) const;
    int getBestTargetID(const cv::Point2f& aim_point) const;
    bool hasTarget() const;
    
    // 获取所有轨迹 (用于可视化)
    const std::vector<Track>& getTracks() const { return tracks_; }

private:
    std::vector<Track> tracks_;
    int next_id_;
    
    // 跟踪器参数
    int max_lost_frames_;       // 允许丢失的最大帧数
    int min_hit_frames_;        // 至少连续命中多少帧才确认为稳定目标
    float max_match_distance_;  // 数据关联的最大距离阈值（像素）
    
    // 全局噪声参数模板
    cv::Mat R_;
    cv::Mat Q_ca_;
    cv::Mat Q_ctrv_;
    
    // 数据关联算法
    void associate(const std::vector<cv::Point2f>& measurements,
                   std::vector<int>& matched_measurements,
                   std::vector<int>& unmatched_tracks,
                   std::vector<int>& unmatched_measurements);
                   
    static double wrapAngle(double angle);
};