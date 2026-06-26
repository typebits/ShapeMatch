#define _USE_MATH_DEFINES
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <immintrin.h>
#include <omp.h>
#include <mutex>

struct Points {
    float dx, dy;   // 相对质心的坐标
    float u, v;     // 归一化梯度方向向量
    float score;
    float angle;
};

std::vector<Points> findAllMatches(const cv::Mat& tgxf, const cv::Mat& tgyf,
    const std::vector<Points>& modelPoints, float minScore, float minDist);

void downsample2x2_simd(const cv::Mat& src, cv::Mat& dst);

float PointScore(const cv::Mat& tgx, const cv::Mat& tgy,
    const std::vector<Points>& instPoints, int r, int c);

std::vector<Points> matchModelPyramidN(
    std::vector<cv::Mat>& pyramidGx,
    std::vector<cv::Mat>& pyramidGy,
    const cv::Mat& topGxF,
    const cv::Mat& topGyF,
    const std::vector<std::vector<Points>>& pyramidModels,
    int numLevels
);

void applyNMS(std::vector<Points>& results, float minDist, float minAngleDist = -1.0f);

void drawPoints(cv::Mat& img_target,
    const std::vector<Points>& allResults,
    const std::vector<Points>& baseModelPoints,
    float minScore);