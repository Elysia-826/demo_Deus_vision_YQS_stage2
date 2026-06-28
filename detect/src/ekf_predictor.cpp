#include "ekf_predictor.h"
#include <limits>
#include <algorithm>
// 大二同学自行实现
EKFPredictor::EKFPredictor() 
    : next_id_(1), max_lost_frames_(30), min_hit_frames_(3), max_match_distance_(200.0f) {
    // 初始化全局噪声模板
    R_ = cv::Mat::eye(2, 2, CV_64F) * 0.08;
    
    Q_ca_ = cv::Mat::eye(6, 6, CV_64F) * 0.05;
    
    Q_ctrv_ = cv::Mat::eye(5, 5, CV_64F);
    Q_ctrv_.at<double>(0, 0) = 0.05; Q_ctrv_.at<double>(1, 1) = 0.05;
    Q_ctrv_.at<double>(2, 2) = 0.10; Q_ctrv_.at<double>(3, 3) = 0.30;
    Q_ctrv_.at<double>(4, 4) = 0.50;
}

// ================= Track 内部实现 =================
void EKFPredictor::Track::initTrack(const cv::Point2f& initial_pos, const cv::Mat& R_mat, const cv::Mat& Q_ca_mat, const cv::Mat& Q_ctrv_mat) {
    current_model = CA;
    is_initialized = true;
    R = R_mat; Q_ca = Q_ca_mat; Q_ctrv = Q_ctrv_mat;
    
    x_ca = cv::Mat::zeros(6, 1, CV_64F);
    P_ca = cv::Mat::eye(6, 6, CV_64F) * 1.0;
    F_ca = cv::Mat::eye(6, 6, CV_64F);
    H_ca = cv::Mat::zeros(2, 6, CV_64F);
    H_ca.at<double>(0, 0) = 1; H_ca.at<double>(1, 1) = 1;
    
    x_ctrv = cv::Mat::zeros(5, 1, CV_64F);
    P_ctrv = cv::Mat::eye(5, 5, CV_64F) * 1.0;
    H_ctrv = cv::Mat::zeros(2, 5, CV_64F);
    H_ctrv.at<double>(0, 0) = 1; H_ctrv.at<double>(1, 1) = 1;
    
    x_ca.at<double>(0) = initial_pos.x; x_ca.at<double>(1) = initial_pos.y;
    x_ctrv.at<double>(0) = initial_pos.x; x_ctrv.at<double>(1) = initial_pos.y;
    
    smoothed_pos = initial_pos;
    lost_frames = 0; hit_frames = 1; is_confirmed = false;
}

cv::Point2f EKFPredictor::Track::predictTrack(double dt) {
    if (!is_initialized) return cv::Point2f(0, 0);
    if (dt <= 0 || dt > 0.1) dt = 0.033;
    
    // 自动热切换逻辑
    if (shouldSwitchToCTRV()) { caToCtrv(); current_model = CTRV; }
    else if (shouldSwitchToCA()) { ctrvToCa(); current_model = CA; }

    if (current_model == CA) predictCA(dt);
    else predictCTRV(dt);
    
    smoothed_pos = getPosition();
    return smoothed_pos;
}

cv::Point2f EKFPredictor::Track::updateTrack(const cv::Point2f& measurement) {
    if (!is_initialized) return measurement;
    cv::Mat z = (cv::Mat_<double>(2, 1) << measurement.x, measurement.y);
    if (current_model == CA) updateCommon(z, x_ca, P_ca, H_ca, R);
    else {
        updateCommon(z, x_ctrv, P_ctrv, H_ctrv, R);
        x_ctrv.at<double>(3) = wrapAngle(x_ctrv.at<double>(3));
    }
    smoothed_pos = getPosition();
    return smoothed_pos;
}

cv::Point2f EKFPredictor::Track::getPosition() const {
    if (current_model == CA) return cv::Point2f(x_ca.at<double>(0), x_ca.at<double>(1));
    return cv::Point2f(x_ctrv.at<double>(0), x_ctrv.at<double>(1));
}

void EKFPredictor::Track::predictCA(double dt) {
    F_ca = cv::Mat::eye(6, 6, CV_64F);
    for (int i = 0; i < 2; ++i) {
        F_ca.at<double>(i, i + 2) = dt; F_ca.at<double>(i, i + 4) = 0.5 * dt * dt;
        F_ca.at<double>(i + 2, i + 4) = dt;
    }
    x_ca = F_ca * x_ca; P_ca = F_ca * P_ca * F_ca.t() + Q_ca;
}

void EKFPredictor::Track::predictCTRV(double dt) {
    double x = x_ctrv.at<double>(0), y = x_ctrv.at<double>(1);
    double v = x_ctrv.at<double>(2), yaw = x_ctrv.at<double>(3), omega = x_ctrv.at<double>(4);
    cv::Mat x_pred = cv::Mat::zeros(5, 1, CV_64F);
    cv::Mat F = cv::Mat::eye(5, 5, CV_64F);
    if (std::abs(omega) > 1e-3) {
        double sin_yaw = std::sin(yaw), cos_yaw = std::cos(yaw);
        double sin_yaw_next = std::sin(yaw + omega * dt), cos_yaw_next = std::cos(yaw + omega * dt);
        x_pred.at<double>(0) = x + v / omega * (sin_yaw_next - sin_yaw);
        x_pred.at<double>(1) = y + v / omega * (cos_yaw - cos_yaw_next);
        x_pred.at<double>(2) = v; x_pred.at<double>(3) = wrapAngle(yaw + omega * dt); x_pred.at<double>(4) = omega;
        F.at<double>(0, 2) = (sin_yaw_next - sin_yaw) / omega; F.at<double>(0, 3) = v / omega * (cos_yaw_next - cos_yaw);
        F.at<double>(0, 4) = v * dt / omega * cos_yaw_next - v / (omega * omega) * (sin_yaw_next - sin_yaw);
        F.at<double>(1, 2) = (cos_yaw - cos_yaw_next) / omega; F.at<double>(1, 3) = v / omega * (sin_yaw_next - sin_yaw);
        F.at<double>(1, 4) = v * dt / omega * sin_yaw_next - v / (omega * omega) * (cos_yaw - cos_yaw_next);
        F.at<double>(3, 4) = dt;
    } else {
        x_pred.at<double>(0) = x + v * std::cos(yaw) * dt; x_pred.at<double>(1) = y + v * std::sin(yaw) * dt;
        x_pred.at<double>(2) = v; x_pred.at<double>(3) = yaw; x_pred.at<double>(4) = omega;
        F.at<double>(0, 2) = std::cos(yaw) * dt; F.at<double>(0, 3) = -v * std::sin(yaw) * dt;
        F.at<double>(1, 2) = std::sin(yaw) * dt; F.at<double>(1, 3) = v * std::cos(yaw) * dt;
    }
    x_ctrv = x_pred; P_ctrv = F * P_ctrv * F.t() + Q_ctrv;
}

void EKFPredictor::Track::updateCommon(const cv::Mat& z, cv::Mat& x, cv::Mat& P, const cv::Mat& H, const cv::Mat& R_mat) {
    cv::Mat PHt = P * H.t(); cv::Mat S = H * PHt + R_mat;
    cv::Mat K = PHt * S.inv(cv::DECOMP_LU); cv::Mat y = z - H * x;
    x = x + K * y;
    cv::Mat I = cv::Mat::eye(P.rows, P.rows, CV_64F);
    P = (I - K * H) * P;
}

void EKFPredictor::Track::caToCtrv() {
    double vx = x_ca.at<double>(2), vy = x_ca.at<double>(3);
    x_ctrv.at<double>(0) = x_ca.at<double>(0); x_ctrv.at<double>(1) = x_ca.at<double>(1);
    x_ctrv.at<double>(2) = std::sqrt(vx*vx + vy*vy); x_ctrv.at<double>(3) = std::atan2(vy, vx); x_ctrv.at<double>(4) = 0.0;
    P_ctrv = cv::Mat::eye(5, 5, CV_64F) * 0.5;
    P_ctrv.at<double>(0, 0) = P_ca.at<double>(0, 0); P_ctrv.at<double>(1, 1) = P_ca.at<double>(1, 1);
}

void EKFPredictor::Track::ctrvToCa() {
    double v = x_ctrv.at<double>(2), yaw = x_ctrv.at<double>(3);
    x_ca.at<double>(0) = x_ctrv.at<double>(0); x_ca.at<double>(1) = x_ctrv.at<double>(1);
    x_ca.at<double>(2) = v * std::cos(yaw); x_ca.at<double>(3) = v * std::sin(yaw);
    x_ca.at<double>(4) = 0.0; x_ca.at<double>(5) = 0.0;
    P_ca = cv::Mat::eye(6, 6, CV_64F) * 0.5;
    P_ca.at<double>(0, 0) = P_ctrv.at<double>(0, 0); P_ca.at<double>(1, 1) = P_ctrv.at<double>(1, 1);
}

bool EKFPredictor::Track::shouldSwitchToCTRV() const {
    if (current_model == CTRV) return false;
    double v = std::sqrt(x_ca.at<double>(2)*x_ca.at<double>(2) + x_ca.at<double>(3)*x_ca.at<double>(3));
    double a = std::sqrt(x_ca.at<double>(4)*x_ca.at<double>(4) + x_ca.at<double>(5)*x_ca.at<double>(5));
    return (v < 1.5 && a > 0.8);
}

bool EKFPredictor::Track::shouldSwitchToCA() const {
    if (current_model == CA) return false;
    double omega = std::abs(x_ctrv.at<double>(4)); double v = x_ctrv.at<double>(2);
    return (omega < 0.5 && v > 0.5);
}

// ================= EKFPredictor 多目标管理实现 =================
void EKFPredictor::update(const std::vector<cv::Point2f>& measurements, double dt) {
    for (auto& track : tracks_) track.predictTrack(dt);

    std::vector<int> matched_measurements(tracks_.size(), -1);
    std::vector<int> unmatched_tracks, unmatched_measurements;
    associate(measurements, matched_measurements, unmatched_tracks, unmatched_measurements);

    for (size_t i = 0; i < tracks_.size(); ++i) {
        int meas_idx = matched_measurements[i];
        if (meas_idx != -1) {
            tracks_[i].updateTrack(measurements[meas_idx]);
            tracks_[i].lost_frames = 0;
            tracks_[i].hit_frames++;
            if (tracks_[i].hit_frames >= min_hit_frames_) tracks_[i].is_confirmed = true;
        } else {
            tracks_[i].lost_frames++;
        }
    }

    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), 
        [this](const Track& t) { return t.lost_frames > max_lost_frames_; }), tracks_.end());

    for (int meas_idx : unmatched_measurements) {
        Track new_track;
        new_track.id = next_id_++;
        new_track.initTrack(measurements[meas_idx], R_, Q_ca_, Q_ctrv_);
        tracks_.push_back(new_track);
    }
}

void EKFPredictor::associate(const std::vector<cv::Point2f>& measurements, std::vector<int>& matched_measurements, std::vector<int>& unmatched_tracks, std::vector<int>& unmatched_measurements) {
    int num_tracks = tracks_.size(), num_meas = measurements.size();
    matched_measurements.assign(num_tracks, -1);
    std::vector<bool> meas_matched(num_meas, false);

    if (num_tracks == 0) { for (int i = 0; i < num_meas; ++i) unmatched_measurements.push_back(i); return; }
    if (num_meas == 0) { for (int i = 0; i < num_tracks; ++i) unmatched_tracks.push_back(i); return; }

    std::vector<std::vector<float>> cost_matrix(num_tracks, std::vector<float>(num_meas, 0.0f));
    for (int i = 0; i < num_tracks; ++i)
        for (int j = 0; j < num_meas; ++j)
            cost_matrix[i][j] = cv::norm(tracks_[i].smoothed_pos - measurements[j]);

    struct MatchPair { int t_idx, m_idx; float cost; bool operator<(const MatchPair& o) const { return cost < o.cost; } };
    std::vector<MatchPair> pairs;
    for (int i = 0; i < num_tracks; ++i)
        for (int j = 0; j < num_meas; ++j)
            if (cost_matrix[i][j] < max_match_distance_) pairs.push_back({i, j, cost_matrix[i][j]});

    std::sort(pairs.begin(), pairs.end());
    std::vector<bool> track_matched(num_tracks, false);
    for (const auto& p : pairs) {
        if (!track_matched[p.t_idx] && !meas_matched[p.m_idx]) {
            matched_measurements[p.t_idx] = p.m_idx;
            track_matched[p.t_idx] = true; meas_matched[p.m_idx] = true;
        }
    }
    for (int i = 0; i < num_tracks; ++i) if (!track_matched[i]) unmatched_tracks.push_back(i);
    for (int j = 0; j < num_meas; ++j) if (!meas_matched[j]) unmatched_measurements.push_back(j);
}

cv::Point2f EKFPredictor::getBestTarget(const cv::Point2f& aim_point) const {
    int best_id = getBestTargetID(aim_point);
    for (const auto& t : tracks_) if (t.id == best_id) return t.smoothed_pos;
    return cv::Point2f(0, 0);
}

int EKFPredictor::getBestTargetID(const cv::Point2f& aim_point) const {
    int best_id = -1; float min_dist = std::numeric_limits<float>::max();
    for (const auto& t : tracks_) {
        if (!t.is_confirmed) continue;
        float dist = cv::norm(t.smoothed_pos - aim_point);
        if (dist < min_dist) { min_dist = dist; best_id = t.id; }
    }
    return best_id;
}

bool EKFPredictor::hasTarget() const {
    for (const auto& t : tracks_) if (t.is_confirmed) return true;
    return false;
}

double EKFPredictor::wrapAngle(double angle) {
    while (angle > CV_PI) angle -= 2 * CV_PI;
    while (angle < -CV_PI) angle += 2 * CV_PI;
    return angle;
}