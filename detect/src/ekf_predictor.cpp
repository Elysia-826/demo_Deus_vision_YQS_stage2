#include "ekf_predictor.h"

// 大二同学自行实现
EKFPredictor::EKFPredictor() 
    : current_model_(CA), is_initialized_(false) {
    
    // === 公共参数 ===
    R_ = cv::Mat::eye(2, 2, CV_64F) * 0.08; // 观测噪声 (x,y)

    // === CA 模型初始化 (6维) ===
    x_ca_ = cv::Mat::zeros(6, 1, CV_64F);
    P_ca_ = cv::Mat::eye(6, 6, CV_64F) * 1.0;
    F_ca_ = cv::Mat::eye(6, 6, CV_64F);
    H_ca_ = cv::Mat::zeros(2, 6, CV_64F);
    H_ca_.at<double>(0, 0) = 1; H_ca_.at<double>(1, 1) = 1;
    Q_ca_ = cv::Mat::eye(6, 6, CV_64F) * 0.05; // 平移过程噪声

    // === CTRV 模型初始化 (5维) ===
    x_ctrv_ = cv::Mat::zeros(5, 1, CV_64F);
    P_ctrv_ = cv::Mat::eye(5, 5, CV_64F) * 1.0;
    H_ctrv_ = cv::Mat::zeros(2, 5, CV_64F);
    H_ctrv_.at<double>(0, 0) = 1; H_ctrv_.at<double>(1, 1) = 1;
    
    // 小陀螺特化：提高 yaw 和 omega 的过程噪声，允许快速旋转变化
    Q_ctrv_ = cv::Mat::eye(5, 5, CV_64F);
    Q_ctrv_.at<double>(0, 0) = 0.05; // x
    Q_ctrv_.at<double>(1, 1) = 0.05; // y
    Q_ctrv_.at<double>(2, 2) = 0.10; // v (小陀螺线速度通常较小且波动)
    Q_ctrv_.at<double>(3, 3) = 0.30; // yaw (允许航向快速变化)
    Q_ctrv_.at<double>(4, 4) = 0.50; // omega (允许角速度剧烈变化)
}

void EKFPredictor::init(const cv::Point2f& initial_pos) {
    x_ca_.at<double>(0) = initial_pos.x;
    x_ca_.at<double>(1) = initial_pos.y;
    x_ctrv_.at<double>(0) = initial_pos.x;
    x_ctrv_.at<double>(1) = initial_pos.y;
    is_initialized_ = true;
}

cv::Point2f EKFPredictor::predict(double dt) {
    if (!is_initialized_) return cv::Point2f(0, 0);
    if (dt <= 0 || dt > 0.1) dt = 0.033; // 安全钳制 dt

    if (current_model_ == CA) predictCA(dt);
    else predictCTRV(dt);

    return getPosition();
}

cv::Point2f EKFPredictor::update(const cv::Point2f& measurement) {
    if (!is_initialized_) { init(measurement); return measurement; }

    cv::Mat z = (cv::Mat_<double>(2, 1) << measurement.x, measurement.y);
    
    if (current_model_ == CA) {
        updateCommon(z, x_ca_, P_ca_, H_ca_, R_);
    } else {
        updateCommon(z, x_ctrv_, P_ctrv_, H_ctrv_, R_);
        // 🌀 小陀螺特化：更新后强制包裹 yaw 角，防止数值溢出
        x_ctrv_.at<double>(3) = wrapAngle(x_ctrv_.at<double>(3));
    }
    return getPosition();
}

void EKFPredictor::predictCA(double dt) {
    // 更新 F 矩阵 (匀加速模型)
    F_ca_ = cv::Mat::eye(6, 6, CV_64F);
    for (int i = 0; i < 2; ++i) {
        F_ca_.at<double>(i, i + 2) = dt;
        F_ca_.at<double>(i, i + 4) = 0.5 * dt * dt;
        F_ca_.at<double>(i + 2, i + 4) = dt;
    }
    x_ca_ = F_ca_ * x_ca_;
    P_ca_ = F_ca_ * P_ca_ * F_ca_.t() + Q_ca_;
}

void EKFPredictor::predictCTRV(double dt) {
    double x = x_ctrv_.at<double>(0), y = x_ctrv_.at<double>(1);
    double v = x_ctrv_.at<double>(2), yaw = x_ctrv_.at<double>(3), omega = x_ctrv_.at<double>(4);

    cv::Mat x_pred = cv::Mat::zeros(5, 1, CV_64F);
    cv::Mat F = cv::Mat::eye(5, 5, CV_64F); // 雅可比近似矩阵

    if (std::abs(omega) > 1e-3) {
        // 旋转运动 (含小陀螺高 omega 场景)
        double sin_yaw = std::sin(yaw), cos_yaw = std::cos(yaw);
        double sin_yaw_next = std::sin(yaw + omega * dt);
        double cos_yaw_next = std::cos(yaw + omega * dt);
        
        x_pred.at<double>(0) = x + v / omega * (sin_yaw_next - sin_yaw);
        x_pred.at<double>(1) = y + v / omega * (cos_yaw - cos_yaw_next);
        x_pred.at<double>(2) = v;
        x_pred.at<double>(3) = wrapAngle(yaw + omega * dt);
        x_pred.at<double>(4) = omega;

        // 雅可比矩阵 F (用于协方差传播)
        F.at<double>(0, 2) = (sin_yaw_next - sin_yaw) / omega;
        F.at<double>(0, 3) = v / omega * (cos_yaw_next - cos_yaw);
        F.at<double>(0, 4) = v * dt / omega * cos_yaw_next - v / (omega * omega) * (sin_yaw_next - sin_yaw);
        F.at<double>(1, 2) = (cos_yaw - cos_yaw_next) / omega;
        F.at<double>(1, 3) = v / omega * (sin_yaw_next - sin_yaw);
        F.at<double>(1, 4) = v * dt / omega * sin_yaw_next - v / (omega * omega) * (cos_yaw - cos_yaw_next);
        F.at<double>(3, 4) = dt;
    } else {
        // 直线运动 (omega ≈ 0)
        x_pred.at<double>(0) = x + v * std::cos(yaw) * dt;
        x_pred.at<double>(1) = y + v * std::sin(yaw) * dt;
        x_pred.at<double>(2) = v;
        x_pred.at<double>(3) = yaw;
        x_pred.at<double>(4) = omega;

        F.at<double>(0, 2) = std::cos(yaw) * dt;
        F.at<double>(0, 3) = -v * std::sin(yaw) * dt;
        F.at<double>(1, 2) = std::sin(yaw) * dt;
        F.at<double>(1, 3) = v * std::cos(yaw) * dt;
    }

    x_ctrv_ = x_pred;
    P_ctrv_ = F * P_ctrv_ * F.t() + Q_ctrv_;
}

void EKFPredictor::updateCommon(const cv::Mat& z, cv::Mat& x, cv::Mat& P, const cv::Mat& H, const cv::Mat& R) {
    cv::Mat PHt = P * H.t();
    cv::Mat S = H * PHt + R;
    cv::Mat K = PHt * S.inv(cv::DECOMP_LU);
    cv::Mat y = z - H * x;
    x = x + K * y;
    cv::Mat I = cv::Mat::eye(P.rows, P.rows, CV_64F);
    P = (I - K * H) * P;
}

// 热切换核心：状态映射 + 协方差安全重置
void EKFPredictor::switchTo(ModelType target) {
    if (target == current_model_) return;

    if (target == CTRV) caToCtrv();
    else ctrvToCa();
    
    current_model_ = target;
    std::cout << "[EKF] Hot-switched to " << (target == CTRV ? "CTRV (Spinning)" : "CA (Linear)") << std::endl;
}

void EKFPredictor::caToCtrv() {
    double vx = x_ca_.at<double>(2), vy = x_ca_.at<double>(3);
    x_ctrv_.at<double>(0) = x_ca_.at<double>(0); // x
    x_ctrv_.at<double>(1) = x_ca_.at<double>(1); // y
    x_ctrv_.at<double>(2) = std::sqrt(vx*vx + vy*vy); // v
    x_ctrv_.at<double>(3) = std::atan2(vy, vx);       // yaw
    x_ctrv_.at<double>(4) = 0.0; // omega 初始为0，滤波器会快速自适应
    
    // 协方差重置：保留位置不确定性，重置动态量以防发散
    P_ctrv_ = cv::Mat::eye(5, 5, CV_64F) * 0.5;
    P_ctrv_.at<double>(0, 0) = P_ca_.at<double>(0, 0);
    P_ctrv_.at<double>(1, 1) = P_ca_.at<double>(1, 1);
}

void EKFPredictor::ctrvToCa() {
    double v = x_ctrv_.at<double>(2), yaw = x_ctrv_.at<double>(3);
    x_ca_.at<double>(0) = x_ctrv_.at<double>(0);
    x_ca_.at<double>(1) = x_ctrv_.at<double>(1);
    x_ca_.at<double>(2) = v * std::cos(yaw); // vx
    x_ca_.at<double>(3) = v * std::sin(yaw); // vy
    x_ca_.at<double>(4) = 0.0; // ax
    x_ca_.at<double>(5) = 0.0; // ay
    
    P_ca_ = cv::Mat::eye(6, 6, CV_64F) * 0.5;
    P_ca_.at<double>(0, 0) = P_ctrv_.at<double>(0, 0);
    P_ca_.at<double>(1, 1) = P_ctrv_.at<double>(1, 1);
}

double EKFPredictor::wrapAngle(double angle) {
    while (angle > CV_PI) angle -= 2 * CV_PI;
    while (angle < -CV_PI) angle += 2 * CV_PI;
    return angle;
}

cv::Point2f EKFPredictor::getPosition() const {
    if (current_model_ == CA) return cv::Point2f(x_ca_.at<double>(0), x_ca_.at<double>(1));
    return cv::Point2f(x_ctrv_.at<double>(0), x_ctrv_.at<double>(1));
}

cv::Point2f EKFPredictor::getVelocity() const {
    if (current_model_ == CA) return cv::Point2f(x_ca_.at<double>(2), x_ca_.at<double>(3));
    double v = x_ctrv_.at<double>(2), yaw = x_ctrv_.at<double>(3);
    return cv::Point2f(v * std::cos(yaw), v * std::sin(yaw));
}

double EKFPredictor::getAngularRate() const {
    return (current_model_ == CTRV) ? x_ctrv_.at<double>(4) : 0.0;
}

// 自动切换启发式规则
bool EKFPredictor::shouldSwitchToCTRV() const {
    if (current_model_ == CTRV) return false;
    double v = std::sqrt(x_ca_.at<double>(2)*x_ca_.at<double>(2) + x_ca_.at<double>(3)*x_ca_.at<double>(3));
    // 小陀螺特征：线速度较低，但加速度/转向趋势明显（此处用加速度模长近似曲率）
    double a = std::sqrt(x_ca_.at<double>(4)*x_ca_.at<double>(4) + x_ca_.at<double>(5)*x_ca_.at<double>(5));
    return (v < 1.5 && a > 0.8); // 阈值可根据实车调整
}

bool EKFPredictor::shouldSwitchToCA() const {
    if (current_model_ == CA) return false;
    double omega = std::abs(x_ctrv_.at<double>(4));
    double v = x_ctrv_.at<double>(2);
    return (omega < 0.5 && v > 0.5); // 角速度下降且恢复直线运动
}