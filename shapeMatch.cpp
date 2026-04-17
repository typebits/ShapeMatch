#include "shapeMatch.h"

std::vector<Points> findAllMatches(const cv::Mat& tgx, const cv::Mat& tgy,
    const std::vector<Points>& modelPoints,
    float minScore, float minDist) {

    std::vector<Points> allCandidates;
    unsigned int rows = tgx.rows;
    unsigned int cols = tgx.cols;
    const size_t nPts = modelPoints.size();

    if (nPts == 0) return allCandidates;

    std::vector<const short*> pX(rows), pY(rows);
    for (int i = 0; i < rows; ++i) {
        pX[i] = tgx.ptr<short>(i);
        pY[i] = tgy.ptr<short>(i);
    }

    unsigned int error_p = (uint)nPts / 20.0f;
    unsigned int step = 1; 

    for (int r = 0; r < rows; ++r ) {
        for (int c = 0; c < cols; ++c) {
            float totalDirectionScore = 0;
            unsigned int validCount = 0;

            for (size_t m = 0; m < nPts; m++) {
                // 计算当前点在图中的绝对坐标
                float curX = (float)c + modelPoints[m].dx;
                float curY = (float)r + modelPoints[m].dy;

                // 快速边界检查
                unsigned int x0 = static_cast<int>(curX);
                unsigned int y0 = static_cast<int>(curY);

                if (x0 >= 0 && x0 < cols - 1 && y0 >= 0 && y0 < rows - 1) {
                    int x1 = x0 + 1;
                    int y1 = y0 + 1;

                    //float sx = (pX[y0][x0] + pX[y0][x1] + pX[y1][x0] + pX[y1][x1]) * 0.25f;
                    //float sy = (pY[y0][x0] + pY[y0][x1] + pY[y1][x0] + pY[y1][x1]) * 0.25f;

                    float sx = (pX[y0][x0] + pX[y1][x1]) >> 1;
                    float sy = (pY[y0][x0] + pY[y1][x1]) >> 1;


                    float magSq = sx * sx + sy * sy;
                    if (magSq > 225.0f) {
                        float invMag = 1.0f / std::sqrt(magSq);

                        // 点积累加
                        float dot = (modelPoints[m].u * sx + modelPoints[m].v * sy) * invMag;
                        totalDirectionScore += dot;
                        validCount++;
                    }
                }

                // 剪枝
                if (m == error_p) {
                    if (validCount < error_p) break;
                    if (totalDirectionScore < minScore) break;
                }
            }

            // 4. 判定匹配
            if (validCount > nPts * 0.05f) {
                float score = totalDirectionScore / nPts;
                if (score >= minScore) {
                    Points matchResult;
                    matchResult.dx = (float)c;
                    matchResult.dy = (float)r;
                    matchResult.score = score;
                    allCandidates.push_back(matchResult);
                }
            }
        }
    }

    // 5. NMS 处理
    if (allCandidates.size() > 1) {
        applyNMS(allCandidates, minDist);
    }

    return allCandidates;
}

void downsample2x2(const cv::Mat& src, cv::Mat& dst) {
    int dstRows = src.rows >> 1;
    int dstCols = src.cols >> 1;
    dst.create(dstRows, dstCols, src.type());

    for (int r = 0; r < dstRows; ++r) {
        const short* s1 = src.ptr<short>(2 * r);
        const short* s2 = src.ptr<short>(2 * r + 1);
        short* d = dst.ptr<short>(r);

        for (int c = 0; c < dstCols; ++c) {
            *d++ = (s1[0] + s1[1] + s2[0] + s2[1]) >> 2;// 除以4
            s1 += 2;
            s2 += 2;
        }
    }
}

// 相似匹配
float PointScore(const cv::Mat& tgx, const cv::Mat& tgy,
    const std::vector<Points>& instPoints,
    int r, int c) {
    int validCount = 0;
    float totalCosineScore = 0.0f;

    const short* pGdxBase = tgx.ptr<short>(0);
    const short* pGdyBase = tgy.ptr<short>(0);
    int step = static_cast<int>(tgx.step / sizeof(short));
    int rows = tgx.rows;
    int cols = tgx.cols;

    for (size_t m = 0; m < instPoints.size(); m++) {
        const auto& pt = instPoints[m];

        int curY = r + static_cast<int>(pt.dy + (pt.dy >= 0 ? 0.5f : -0.5f));
        int curX = c + static_cast<int>(pt.dx + (pt.dx >= 0 ? 0.5f : -0.5f));

        if (static_cast<unsigned>(curY) < static_cast<unsigned>(rows) &&
            static_cast<unsigned>(curX) < static_cast<unsigned>(cols)) {

            int offset = curY * step + curX;
            float gxS = static_cast<float>(pGdxBase[offset]);
            float gyS = static_cast<float>(pGdyBase[offset]);

            float magSqS = gxS * gxS + gyS * gyS;

            if (magSqS > 0.1f) {
                float gxT = static_cast<float>(pt.u);
                float gyT = static_cast<float>(pt.v);
                float magSqT = gxT * gxT + gyT * gyT;

                if (magSqT > 0.1f) {
                    float dotProduct = gxS * gxT + gyS * gyT;
                    totalCosineScore += dotProduct / std::sqrt(magSqS * magSqT);
                    validCount++;
                }
            }
        }
    }

    if (validCount < instPoints.size() * 0.1) return 0.0f;

    return (validCount > 0) ? std::abs(totalCosineScore / validCount) : 0.0f;
}

std::vector<Points> matchModelPyramidN(
    std::vector<cv::Mat>& pyramidGx,
    std::vector<cv::Mat>& pyramidGy,
    std::vector<std::vector<Points>> pyramidModels,
    int numLevels
) {
    if (numLevels < 1) return {};

    for (int i = 1; i < numLevels; ++i) {
        downsample2x2(pyramidGx[i - 1], pyramidGx[i]);
        downsample2x2(pyramidGy[i - 1], pyramidGy[i]);

        float factor = std::pow(2.0f, (float)i);
        for (const auto& pt0 : pyramidModels[0]) {
            Points lowPt = pt0;
            lowPt.dx /= factor;
            lowPt.dy /= factor;
            pyramidModels[i].push_back(lowPt);
        }
    }

    int topIdx = numLevels - 1;
     
    // 寻找顶层的粗特征
    std::vector<Points> LayerResults = findAllMatches(
        pyramidGx[topIdx], pyramidGy[topIdx],
        pyramidModels[topIdx],
        0.4f, // 匹配分数
        4.0f // 距离
    );

    for (int level = topIdx - 1; level >= 0; --level) {
        std::vector<Points> nextResults;
        nextResults.reserve(LayerResults.size() + LayerResults.size());

        const auto& curGx = pyramidGx[level];
        const auto& curGy = pyramidGy[level];
        const auto& curModel = pyramidModels[level];
        int rows = curGx.rows;
        int cols = curGx.cols;

        float currentThreshold = 0.55f;

        for (const auto& prevRes : LayerResults) {
            int cx = static_cast<int>(prevRes.dx + prevRes.dx + 0.5f);
            int cy = static_cast<int>(prevRes.dy + prevRes.dy + 0.5f);

            int rStart = std::max(0, cy - 2);
            int rEnd = std::min(rows - 1, cy + 2);
            int cStart = std::max(0, cx - 2);
            int cEnd = std::min(cols - 1, cx + 2);

            Points bestLocal = { };

            for (int r = rStart; r <= rEnd; ++r) {
                for (int c = cStart; c <= cEnd; ++c) {
                    float score = PointScore(curGx, curGy, curModel, r, c);

                    if (score > bestLocal.score) {
                        bestLocal.score = score;
                        bestLocal.dx = static_cast<float>(c);
                        bestLocal.dy = static_cast<float>(r);
                    }
                }
            }

            if (bestLocal.score >= currentThreshold) {
                nextResults.push_back(bestLocal);
            }
        }

        if (nextResults.empty()) return {};

        LayerResults = std::move(nextResults);
    }

    return LayerResults;
}

void applyNMS(std::vector<Points>& results, float minDist, float minAngleDist) {
    if (results.size() <= 1) return;

    std::sort(results.begin(), results.end(), [](const Points& a, const Points& b) {
        return a.score > b.score;
    });

    std::vector<Points> kept;
    float thresholdSq = minDist * minDist;

    for (const auto& candidate : results) {
        bool isDuplicate = false;
        for (const auto& confirmed : kept) {

            float dx = candidate.dx - confirmed.dx;
            float dy = candidate.dy - confirmed.dy;
            float distSq = dx * dx + dy * dy;

            
            if (distSq < thresholdSq) {
                // 距离判断
                if (distSq < thresholdSq) {
                    isDuplicate = true;
                    break;
                }

                // 角度判断
                if (minAngleDist < 0) {
                    isDuplicate = true;
                    break;
                }

                float angleDiff = std::abs(candidate.angle - confirmed.angle);
                if (angleDiff > 180.0f) {
                    angleDiff = 360.0f - angleDiff;
                }

                if (angleDiff < minAngleDist) {
                    isDuplicate = true;
                    break;
                }
            }
        }
        if (!isDuplicate) {
            kept.push_back(candidate);
        }
    }
    results = std::move(kept);
}

void drawPoints(cv::Mat& img_target,
    const std::vector<Points>& allResults,
    const std::vector<Points>& baseModelPoints,
    float minScore) {

    for (const auto& result : allResults) {

        if (result.score < minScore) continue;

        // 角度转弧度
        double rad = result.angle * CV_PI / 180.0;
        double cosA = std::cos(rad);
        double sinA = std::sin(rad);

        // 进行旋转和平移变换
        for (const auto& mp : baseModelPoints) {
            // 变换公式：x' = x*cos - y*sin + tx
            int targetX = cvRound(result.dx + mp.dx * cosA - mp.dy * sinA);
            int targetY = cvRound(result.dy + mp.dx * sinA + mp.dy * cosA);

            if (targetX >= 0 && targetX < img_target.cols &&
                targetY >= 0 && targetY < img_target.rows) {
                img_target.at<cv::Vec3b>(targetY, targetX) = cv::Vec3b(0, 255, 0); // 绘制绿色特征点
            }
        }

        // 绘制目标中点
        cv::Point2f center(result.dx, result.dy);
        cv::circle(img_target, center, 5, cv::Scalar(0, 255, 0), -1);

        std::string text = "S:" + std::to_string(result.score).substr(0, 4) +
            " x:" + std::to_string((int)result.dx) +
            " y:" + std::to_string((int)result.dy) +
            " A:" + std::to_string((int)result.angle);
        cv::putText(img_target, text, cv::Point(result.dx + 10, result.dy - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);

        std::cout << "[Match Found] " << text << std::endl;
    }
}