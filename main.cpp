#include <opencv2/opencv.hpp>
#include "shapeMatch.h"
#include "shapeTrain.h"

int main() {
    cv::Mat img_temp = cv::imread("Asset\\temp3.png");
    cv::Mat img_target = cv::imread("Asset\\target3.png");

    if (img_temp.empty() || img_target.empty()) return -1;

    cv::Mat gray, binary;
    cv::cvtColor(img_temp, gray, cv::COLOR_BGRA2GRAY);
    cv::threshold(gray, binary, 80, 255, cv::THRESH_BINARY);
    std::vector<Points> baseModelPoints = extractModelPoints(binary);// 模板

    // 处理目标图片
    cv::Mat target_gx, target_gy, target_gray;
    cv::cvtColor(img_target, target_gray, cv::COLOR_BGRA2GRAY);
    cv::Sobel(target_gray, target_gx, CV_16S, 1, 0, 3);
    cv::Sobel(target_gray, target_gy, CV_16S, 0, 1, 3);

    // 匹配
    float angleStart = 0.0f;
    float angleEnd = 360.0f;
    float angleStep = 12.0f; // 步长
    float Score = 0.8f;

    std::vector<Points> allResults;
    std::vector<std::vector<Points>> templates;

    // 预生成旋转模板
    for (float angle = angleStart; angle < angleEnd; angle += angleStep) {
        auto rotated = rotateModel(baseModelPoints, angle);
        templates.push_back(std::move(rotated));
    }

    // 构建图像金字塔模型
    int Levels = 4;
    std::vector<cv::Mat> pyramidGx(Levels), pyramidGy(Levels);
    std::vector<std::vector<Points>> pyramidModels(Levels);
    pyramidGx[0] = target_gx;
    pyramidGy[0] = target_gy;


    auto start = std::chrono::high_resolution_clock::now(); // 计时

    // 2. 多角度匹配循环
    for (size_t i = 0; i < templates.size() ; ++i) {
        // 初始化 Level 0
        pyramidModels[0] = templates[i];

        auto results = matchModelPyramidN(
            pyramidGx, pyramidGy, pyramidModels, Levels
        );

        for (auto& r : results) {
            r.angle = angleStart + i * angleStep; // 统一赋值角度
            allResults.push_back(r);
        }
    }

    applyNMS(allResults, 30.0f, 5.0f); // 去重

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "匹配目标总耗时: " << elapsed.count() << " ms" << std::endl;

    drawPoints(img_target, allResults, baseModelPoints, Score);

    cv::imshow("模板匹配", img_target);
    cv::waitKey(0);
}