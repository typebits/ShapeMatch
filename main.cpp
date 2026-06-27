#include <opencv2/opencv.hpp>
#include "shapeMatch.h"
#include "shapeTrain.h"

struct ModelPyramid {
    // 存储该角度下所有层级的模板：[Level][Points]
    std::vector<std::vector<Points>> levels;
};

int main() {
    cv::Mat img_temp = cv::imread("Asset\\temp3.png");
    cv::Mat img_target = cv::imread("Asset\\target3.png");

    if (img_temp.empty() || img_target.empty()) {
        std::cout << "图片读取失败！" << std::endl;
        return -1;
    }

    cv::Mat gray, binary;
    if (img_temp.channels() >= 3) {
        cv::cvtColor(img_temp, gray, img_temp.channels() == 4 ? cv::COLOR_BGRA2GRAY : cv::COLOR_BGR2GRAY);
    }
    else {
        gray = img_temp;
    }

    cv::threshold(gray, binary, 80, 255, cv::THRESH_BINARY);
    std::vector<Points> baseModelPoints = extractModelPoints(binary);

    if (baseModelPoints.empty()) {
        std::cout << "错误：未提取到有效的模板特征点！" << std::endl;
        return -1;
    }

    // 处理目标图片梯度
    cv::Mat target_gx, target_gy, target_gray;
    if (img_target.channels() >= 3) {
        cv::cvtColor(img_target, target_gray, img_target.channels() == 4 ? cv::COLOR_BGRA2GRAY : cv::COLOR_BGR2GRAY);
    }
    else {
        target_gray = img_target;
    }

    cv::Sobel(target_gray, target_gx, CV_16S, 1, 0, 3);
    cv::Sobel(target_gray, target_gy, CV_16S, 0, 1, 3);

    // 匹配参数
    float angleStart = 0.0f;
    float angleEnd = 360.0f;
    float angleStep = 12.0f;
    float Score = 0.75f;

    std::vector<Points> allResults;
    std::vector<std::vector<Points>> templates;

    for (float angle = angleStart; angle < angleEnd; angle += angleStep) {
        templates.push_back(rotateModel(baseModelPoints, angle));
    }

    // 构建目标图的图像金字塔模型
    int Levels = 4;
    std::vector<cv::Mat> pyramidGx(Levels), pyramidGy(Levels);
    pyramidGx[0] = target_gx;
    pyramidGy[0] = target_gy;

    for (int i = 1; i < Levels; ++i) {
        downsample2x2_simd(pyramidGx[i - 1], pyramidGx[i]);
        downsample2x2_simd(pyramidGy[i - 1], pyramidGy[i]);
    }

    int topIdx = Levels - 1;
    cv::Mat topGxF, topGyF;
    pyramidGx[topIdx].convertTo(topGxF, CV_32F);
    pyramidGy[topIdx].convertTo(topGyF, CV_32F);
    if (!topGxF.isContinuous()) topGxF = topGxF.clone();
    if (!topGyF.isContinuous()) topGyF = topGyF.clone();

    int maxThreads = omp_get_max_threads();
    std::vector<std::vector<std::vector<Points>>> threadPyramids(maxThreads, std::vector<std::vector<Points>>(Levels));
    std::vector<std::vector<Points>> threadLocalResults(maxThreads);

    // 提前为所有线程缓存 reserve 空间
    size_t maxPointsCount = baseModelPoints.size();
    for (int t = 0; t < maxThreads; ++t) {
        threadLocalResults[t].reserve(32);
        for (int l = 0; l < Levels; ++l) {
            threadPyramids[t][l].reserve(maxPointsCount);
        }
    }

    std::vector<float> invFactors(Levels);
    for (int l = 0; l < Levels; ++l) {
        invFactors[l] = 1.0f / static_cast<float>(1 << l);
    }

    

    // 1. 预计算阶段 (在并行循环外)
    std::vector<ModelPyramid> precomputedModels(templates.size());

    for (size_t i = 0; i < templates.size(); ++i) {
        const auto& rootTemplate = templates[i];
        size_t nPts = rootTemplate.size();

        precomputedModels[i].levels.resize(Levels);
        precomputedModels[i].levels[0] = rootTemplate; // 第0层是原始模板

        for (int l = 1; l < Levels; ++l) {
            float factor = invFactors[l];
            precomputedModels[i].levels[l].resize(nPts);
            for (size_t p = 0; p < nPts; ++p) {
                precomputedModels[i].levels[l][p] = {
                    rootTemplate[p].dx * factor,
                    rootTemplate[p].dy * factor,
                    rootTemplate[p].u,
                    rootTemplate[p].v,
                    0.0f
                };
            }
        }
    }
    auto start = std::chrono::high_resolution_clock::now();

    // 2. 并行匹配阶段 (在 omp parallel 循环内)
    #pragma omp parallel for
    for (int i = 0; i < (int)templates.size(); ++i) {
        int tid = omp_get_thread_num();

        // 直接使用预计算好的模型，避免 resize 和坐标转换
        const auto& modelPyramid = precomputedModels[i].levels;

        // 假设 matchModelPyramidN 支持传入预计算好的模型
        auto results = matchModelPyramidN(pyramidGx, pyramidGy, topGxF, topGyF, modelPyramid, Levels);

        if (!results.empty()) {
            float currentAngle = angleStart + static_cast<float>(i) * angleStep;
            for (auto& res : results) {
                res.angle = currentAngle;
            }
            threadLocalResults[tid].insert(threadLocalResults[tid].end(), results.begin(), results.end());
        }
    }

    for (int t = 0; t < maxThreads; ++t) {
        if (!threadLocalResults[t].empty()) {
            allResults.insert(allResults.end(), threadLocalResults[t].begin(), threadLocalResults[t].end());
        }
    }

    applyNMS(allResults, 30.0f, 5.0f);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "匹配目标总耗时: " << elapsed.count() << " ms" << std::endl;

    // 绘制与展示
    drawPoints(img_target, allResults, baseModelPoints, Score);

    cv::imshow("模板匹配", img_target);
    cv::waitKey(0);
    return 0;
}