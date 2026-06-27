#include "shapeTrain.h"

std::vector<Points> rotateModel(const std::vector<Points>& srcModel, float angleDeg) {
    std::vector<Points> rotatedModel;
    rotatedModel.reserve(srcModel.size());

    // 将角度转换为弧度
    float angleRad = angleDeg * (CV_PI / 180.0f);
    float cosA = std::cos(angleRad);
    float sinA = std::sin(angleRad);

    for (const auto& pt : srcModel) {
        float x = pt.dx, y = pt.dy;
        float u = pt.u, v = pt.v;

        // 每一项只计算一次乘法
        float x_cos = x * cosA, x_sin = x * sinA;
        float y_cos = y * cosA, y_sin = y * sinA;
        float u_cos = u * cosA, u_sin = u * sinA;
        float v_cos = v * cosA, v_sin = v * sinA;

        rotatedModel.push_back({
            x_cos - y_sin, x_sin + y_cos, // dx, dy
            u_cos - v_sin, u_sin + v_cos  // u, v
            });
    }

    return rotatedModel;
}

std::vector<Points> extractModelPoints(const cv::Mat& src) {
    std::vector<Points> modelPoints;

    cv::Mat small_gx, small_gy;

    cv::Sobel(src, small_gx, CV_16S, 1, 0, 3);
    cv::Sobel(src, small_gy, CV_16S, 0, 1, 3);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(src, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    if (contours.empty()) return modelPoints;

    size_t maxIdx = 0;
    double maxArea = 0;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            maxIdx = i;
        }
    }

    const std::vector<cv::Point>& templateContour = contours[maxIdx];

    cv::Moments mu = cv::moments(templateContour);
    if (mu.m00 == 0) return modelPoints; // 防止除以0
    cv::Point2f center(mu.m10 / mu.m00, mu.m01 / mu.m00);

    cv::Mat visual;
    cv::cvtColor(src, visual, cv::COLOR_GRAY2BGR);

    modelPoints.reserve(templateContour.size() / 2);
    for (size_t i = 0; i < templateContour.size(); i += 2) {
        const auto& pt = templateContour[i];
        // 访问梯度值 y 是行，x 是列
        float gx = (float)small_gx.at<short>(pt.y, pt.x);
        float gy = (float)small_gy.at<short>(pt.y, pt.x);

        float magSq = gx * gx + gy * gy;
        if (magSq > 225.0f) {
            float invMag = 1.0f / std::sqrt(magSq);
            Points smp;
            smp.dx = (float)pt.x - center.x;
            smp.dy = (float)pt.y - center.y;
            smp.u = gx * invMag;
            smp.v = gy * invMag;
            modelPoints.push_back(smp);// 归一化

            cv::circle(visual, pt, 2, cv::Scalar(0, 255, 0), -1);
        }
    }

    //cv::imshow("模板轮廓", visual);
    //cv::waitKey(0);

    return modelPoints;
}