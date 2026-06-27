#include "shapeMatch.h"
#include <immintrin.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>

// 保持高效的水平求和
inline float hsum_256_fast(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    vlow = _mm_hadd_ps(vlow, vlow);
    vlow = _mm_hadd_ps(vlow, vlow);
    return _mm_cvtss_f32(vlow);
}

std::vector<Points> findAllMatches(const cv::Mat& tgxf, const cv::Mat& tgyf,
    const std::vector<Points>& modelPoints, float minScore, float minDist) {

    const int rows = tgxf.rows;
    const int cols = tgxf.cols;
    const float* pBaseX = tgxf.ptr<float>(0);
    const float* pBaseY = tgyf.ptr<float>(0);
    const int step = (int)(tgxf.step1());

    const size_t nPts = modelPoints.size();
    float Limit = nPts * 0.03f;
    if (nPts == 0) return {};


    float min_x = modelPoints[0].dx;
    float max_x = min_x;
    float min_y = modelPoints[0].dy;
    float max_y = min_y;

    for (const auto& pt : modelPoints) {
        const float x = pt.dx;
        const float y = pt.dy;
        min_x = (x < min_x) ? x : min_x;
        max_x = (x > max_x) ? x : max_x;
        min_y = (y < min_y) ? y : min_y;
        max_y = (y > max_y) ? y : max_y;
    }
    int minDx = static_cast<int>(std::floor(min_x));
    int maxDx = static_cast<int>(std::ceil(max_x));
    int minDy = static_cast<int>(std::floor(min_y));
    int maxDy = static_cast<int>(std::ceil(max_y));

    int rStart = std::max(0, -minDy);
    int rEnd = std::min(rows - 1, rows - maxDy - 1);
    int cStart = std::max(0, -minDx);
    int cEnd = std::min(cols - 1, cols - maxDx - 1);

    // 确保宽度是 8 的倍数以便于内层循环 SIMD 线性存储
    int width = cEnd - cStart;
    if (width <= 0 || rEnd <= rStart) return {};

    // 创建局部临时画布，用于累加每个像素的得分和有效点计数
    // 使用一维连续数组，保证局部 L1/L2 缓存命中率
    size_t roiSize = (size_t)(rEnd - rStart) * width;
    std::vector<float> totalScoreMap(roiSize, 0.0f);
    std::vector<float> validCountMap(roiSize, 0.0f);
    std::vector<uint8_t> pixelPassed(roiSize, 1); // 用于记录哪些像素通过了第16个点的早期剪枝

    size_t alignedPts = (nPts + 7) & ~7;
    const __m256 vMagThresh = _mm256_set1_ps(225.0f);
    const __m256 vOne = _mm256_set1_ps(1.0f);
    const __m256 v05 = _mm256_set1_ps(0.5f);

    for (size_t m = 0; m < nPts; ++m) {
        float m_u = modelPoints[m].u;
        float m_v = modelPoints[m].v;
        int dx = (int)std::round(modelPoints[m].dx);
        int dy = (int)std::round(modelPoints[m].dy);
        int offA = dy * step + dx;
        int offB = (dy + 1) * step + (dx + 1);

        const __m256 vU = _mm256_set1_ps(m_u);
        const __m256 vV = _mm256_set1_ps(m_v);

        for (int r = rStart; r < rEnd; ++r) {
            int rowBase = r * step;
            int roiRowOffset = (r - rStart) * width;

            const float* pX_A = pBaseX + rowBase + offA;
            const float* pX_B = pBaseX + rowBase + offB;
            const float* pY_A = pBaseY + rowBase + offA;
            const float* pY_B = pBaseY + rowBase + offB;

            float* pDstScore = &totalScoreMap[roiRowOffset];
            float* pDstCount = &validCountMap[roiRowOffset];
            uint8_t* pPassed = &pixelPassed[roiRowOffset];

            int c = 0;
            for (; c <= width - 8; c += 8) {
                // 如果当前 8 个像素此前已被整体或部分剪枝，可以视情况跳过，但为了 SIMD 连贯性这里直接全量算
                __m256 sax = _mm256_loadu_ps(pX_A + cStart + c);
                __m256 sbx = _mm256_loadu_ps(pX_B + cStart + c);
                __m256 sx = _mm256_mul_ps(_mm256_add_ps(sax, sbx), v05);

                __m256 say = _mm256_loadu_ps(pY_A + cStart + c);
                __m256 sby = _mm256_loadu_ps(pY_B + cStart + c);
                __m256 sy = _mm256_mul_ps(_mm256_add_ps(say, sby), v05);

                __m256 magSq = _mm256_mul_ps(sx, sx);
                magSq = _mm256_fmadd_ps(sy, sy, magSq);

                __m256 mask = _mm256_cmp_ps(magSq, vMagThresh, _CMP_GT_OQ);
                __m256 invMag = _mm256_rsqrt_ps(magSq);

                __m256 dot = _mm256_mul_ps(vU, sx);
                dot = _mm256_fmadd_ps(vV, sy, dot);
                dot = _mm256_mul_ps(dot, invMag);

                // 读取历史得分和计数并累加
                __m256 vHistoryScore = _mm256_loadu_ps(pDstScore + c);
                __m256 vHistoryCount = _mm256_loadu_ps(pDstCount + c);

                vHistoryScore = _mm256_add_ps(vHistoryScore, _mm256_and_ps(mask, dot));
                vHistoryCount = _mm256_add_ps(vHistoryCount, _mm256_and_ps(mask, vOne));

                _mm256_storeu_ps(pDstScore + c, vHistoryScore);
                _mm256_storeu_ps(pDstCount + c, vHistoryCount);

                // 【严格恢复剪枝逻辑】如果在第16个特征点，检查并标记未达标的像素
                if (m == 16) {
                    alignas(32) float counts[8];
                    _mm256_store_ps(counts, vHistoryCount);
                    for (int k = 0; k < 8; ++k) {
                        if (counts[k] < Limit) {
                            pPassed[c + k] = 0; // 0 表示未通过
                        }
                    }
                }
            }

            // 处理余数边界像素
            //for (; c < width; ++c) {
            //    if (isPruneStage && !pPassed[c]) continue;

            //    int actualC = cStart + c;
            //    float sx = (pX_A[actualC] + pX_B[actualC]) * 0.5f;
            //    float sy = (pY_A[actualC] + pY_B[actualC]) * 0.5f;
            //    size_t idx = (size_t)roiRowOffset + c;
            //    float magSq = sx * sx + sy * sy;
            //    if (magSq > 225.0f) {
            //        float invMag = 1.0f / std::sqrt(magSq);
            //        float dot = (m_u * sx + m_v * sy) * invMag;
            //        size_t idx = (size_t)roiRowOffset + c;
            //        pDstScore[c] += dot;
            //        pDstCount[c] += 1.0f;
            //    }
            //    if (pixelPassed[idx] && validCountMap[idx] > countLimit) {
            //        pPassed[c] = false;
            //    }
            //}
        }
    }

    std::vector<Points> localCandidates;
    localCandidates.reserve(128); // 预留少量内存，避免初始分配开销

    const float invNPts = 1.0f / nPts;

    for (int r = rStart; r < rEnd; ++r) {
        const int roiRowOffset = (r - rStart) * width;
        const float* pScore = &totalScoreMap[roiRowOffset];
        const float* pCount = &validCountMap[roiRowOffset];
        const uint8_t* pPassed = &pixelPassed[roiRowOffset];

        for (int c = 0; c < width; ++c) {
            // 将所有逻辑条件合并，减少 if 嵌套
            if (pPassed[c] && pCount[c] > Limit) {
                float finalScore = pScore[c] * invNPts;
                if (finalScore >= minScore) {
                    localCandidates.push_back({ (float)(cStart + c), (float)r, 0.0f, 0.0f, finalScore });
                }
            }
        }
    }

    return localCandidates;
}

// SIMD 2x2 下采样
void downsample2x2_simd(const cv::Mat& src, cv::Mat& dst) {
    int dstRows = src.rows >> 1;
    int dstCols = src.cols >> 1;
    dst.create(dstRows, dstCols, CV_16SC1);
    __m256i ones = _mm256_set1_epi16(1);

    for (int r = 0; r < dstRows; ++r) {
        const short* s1 = src.ptr<short>(2 * r);
        const short* s2 = src.ptr<short>(2 * r + 1);
        short* d = dst.ptr<short>(r);

        int c = 0;
        for (; c <= dstCols - 16; c += 16) {
            __m256i row1_0 = _mm256_loadu_si256((const __m256i*)(s1 + 2 * c));
            __m256i row1_1 = _mm256_loadu_si256((const __m256i*)(s1 + 2 * c + 16));
            __m256i row2_0 = _mm256_loadu_si256((const __m256i*)(s2 + 2 * c));
            __m256i row2_1 = _mm256_loadu_si256((const __m256i*)(s2 + 2 * c + 16));

            __m256i vsum0 = _mm256_add_epi16(row1_0, row2_0);
            __m256i vsum1 = _mm256_add_epi16(row1_1, row2_1);
            __m256i hsum0 = _mm256_madd_epi16(vsum0, ones);
            __m256i hsum1 = _mm256_madd_epi16(vsum1, ones);

            __m256i res32_0 = _mm256_srai_epi32(hsum0, 2);
            __m256i res32_1 = _mm256_srai_epi32(hsum1, 2);
            __m256i packed = _mm256_packs_epi32(res32_0, res32_1);
            __m256i final = _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));

            _mm256_storeu_si256((__m256i*)(d + c), final);
        }
        for (; c < dstCols; ++c) {
            d[c] = (short)((s1[2 * c] + s1[2 * c + 1] + s2[2 * c] + s2[2 * c + 1]) >> 2);
        }
    }
}

// 逐像素相似匹配得分函数
float PointScore(const cv::Mat& tgx, const cv::Mat& tgy,
    const std::vector<Points>& instPoints, int r, int c) {
    unsigned int validCount = 0;
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

            unsigned int offset = curY * step + curX;
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
                    ++validCount;
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
    const cv::Mat& topGxF,
    const cv::Mat& topGyF,
    const std::vector<std::vector<Points>>& pyramidModels,
    int numLevels
) {
    if (numLevels < 1) return {};
    unsigned int topIdx = numLevels - 1;

    std::vector<Points> LayerResults = findAllMatches(
        topGxF, topGyF,
        pyramidModels[topIdx],
        0.40f,
        4.0f
    );

    for (int level = topIdx - 1; level >= 0; --level) {
        if (LayerResults.empty()) return {};

        std::vector<Points> nextResults;
        nextResults.reserve(LayerResults.size() * 2);

        const auto& curGx = pyramidGx[level];
        const auto& curGy = pyramidGy[level];
        const auto& curModel = pyramidModels[level];
        const int rows = curGx.rows;
        const int cols = curGx.cols;

        float currentThreshold = (level == 0) ? 0.65f : 0.55f;

        for (const auto& prevRes : LayerResults) {
            const int cx = static_cast<int>(prevRes.dx * 2.0f + 0.5f);
            const int cy = static_cast<int>(prevRes.dy * 2.0f + 0.5f);

            const int rStart = std::max(0, cy - 2);
            const int rEnd = std::min(rows - 1, cy + 2);
            const int cStart = std::max(0, cx - 2);
            const int cEnd = std::min(cols - 1, cx + 2);

            Points bestLocal = { 0.0f, 0.0f, 0.0f, 0.0f, -1.0f };

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
        LayerResults = std::move(nextResults);
    }

    return LayerResults;
}

// 保持现存高效率 NMS 不变
void applyNMS(std::vector<Points>& results, float minDist, float minAngleDist) {
    if (results.size() <= 1) return;

    std::sort(results.begin(), results.end(), [](const Points& a, const Points& b) {
        return a.score > b.score;
        });

    std::vector<Points> kept;
    float thresholdSq = minDist * minDist;

    for (Points& candidate : results) {
        bool isDuplicate = false;
        for (const Points& confirmed : kept) {
            float dx = candidate.dx - confirmed.dx;
            float dy = candidate.dy - confirmed.dy;
            if ((dx * dx + dy * dy) < thresholdSq) {
                isDuplicate = true;
                break;
            }
        }
        if (!isDuplicate) kept.push_back(candidate);
    }
    results = std::move(kept);
}

// 保持现存绘图函数不变
void drawPoints(cv::Mat& img_target, const std::vector<Points>& allResults,
    const std::vector<Points>& baseModelPoints, float minScore) {
    for (const auto& result : allResults) {
        if (result.score < minScore) continue;

        double rad = result.angle * CV_PI / 180.0;
        double cosA = std::cos(rad), sinA = std::sin(rad);

        for (const auto& mp : baseModelPoints) {
            int targetX = cvRound(result.dx + mp.dx * cosA - mp.dy * sinA);
            int targetY = cvRound(result.dy + mp.dx * sinA + mp.dy * cosA);
            if (targetX >= 0 && targetX < img_target.cols && targetY >= 0 && targetY < img_target.rows) {
                cv::circle(img_target, cv::Point(targetX, targetY), 2, cv::Scalar(0, 255, 0), -1);
            }
        }

        cv::circle(img_target, cv::Point2f(result.dx, result.dy), 5, cv::Scalar(0, 255, 0), -1);
        std::string text = "S:" + std::to_string(result.score).substr(0, 4) +
            " x:" + std::to_string((int)result.dx) + " y:" + std::to_string((int)result.dy) + " A:" + std::to_string((int)result.angle);
        cv::putText(img_target, text, cv::Point(result.dx + 10, result.dy - 10), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);
        std::cout << "[Match Found] " << text << std::endl;
    }
}