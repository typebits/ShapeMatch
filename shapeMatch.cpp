#include "shapeMatch.h"

// 辅助函数：快速水平求和 __m256
inline float _mm256_reduce_add_ps(__m256 v) {
    __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
    x128 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    x128 = _mm_add_ss(x128, _mm_shuffle_ps(x128, x128, 0x01));
    return _mm_cvtss_f32(x128);
}

std::vector<Points> findAllMatches(const cv::Mat& tgx, const cv::Mat& tgy,
    std::vector<Points> modelPoints, // 这里改为传值，内部需要排序
    float minScore, float minDist) {

    std::vector<Points> allCandidates;
    const int rows = tgx.rows;
    const int cols = tgx.cols;
    const size_t nPts = modelPoints.size();
    if (nPts == 0) return allCandidates;

    // 1. 核心优化：按特征重要性（模长/权重）排序，确保前 16 个点是最具代表性的
    // 如果没有权重，保持现状，但预处理成 SoA
    size_t alignedPts = (nPts + 7) & ~7;
    std::vector<float> mDX(alignedPts, 0), mDY(alignedPts, 0), mU(alignedPts, 0), mV(alignedPts, 0);
    for (size_t i = 0; i < nPts; ++i) {
        mDX[i] = modelPoints[i].dx; mDY[i] = modelPoints[i].dy;
        mU[i] = modelPoints[i].u;   mV[i] = modelPoints[i].v;
    }

    std::vector<const short*> pX(rows), pY(rows);
    for (int i = 0; i < rows; ++i) {
        pX[i] = tgx.ptr<short>(i); pY[i] = tgy.ptr<short>(i);
    }

    const __m256 vMagThresh = _mm256_set1_ps(225.0f);
    const __m256 vOne = _mm256_set1_ps(1.0f);

    for (int r = 0; r < rows - 1; ++r) {
        for (int c = 0; c < cols - 1; ++c) {
            __m256 vTotalScore = _mm256_setzero_ps();
            __m256 vValidCount = _mm256_setzero_ps();

            const __m256 vC = _mm256_set1_ps((float)c);
            const __m256 vR = _mm256_set1_ps((float)r);

            bool passEarlyStage = true;

            for (size_t m = 0; m < alignedPts; m += 8) {
                // 加载模板 SoA
                __m256 vDX = _mm256_loadu_ps(&mDX[m]);
                __m256 vDY = _mm256_loadu_ps(&mDY[m]);

                __m256i vX0 = _mm256_cvttps_epi32(_mm256_add_ps(vC, vDX));
                __m256i vY0 = _mm256_cvttps_epi32(_mm256_add_ps(vR, vDY));

                alignas(32) int32_t x_idx[8], y_idx[8];
                _mm256_store_si256((__m256i*)x_idx, vX0);
                _mm256_store_si256((__m256i*)y_idx, vY0);

                alignas(32) float sx_f[8], sy_f[8];

                // 标量提取
                for (int i = 0; i < 8; ++i) {
                    int x = x_idx[i]; int y = y_idx[i];
                    if (x >= 0 && x < cols - 1 && y >= 0 && y < rows - 1) {
                        sx_f[i] = (float)((pX[y][x] + pX[y + 1][x + 1]) >> 1);
                        sy_f[i] = (float)((pY[y][x] + pY[y + 1][x + 1]) >> 1);
                    }
                    else {
                        sx_f[i] = 0; sy_f[i] = 0;
                    }
                }

                __m256 vSX = _mm256_load_ps(sx_f);
                __m256 vSY = _mm256_load_ps(sy_f);
                __m256 vMagSq = _mm256_add_ps(_mm256_mul_ps(vSX, vSX), _mm256_mul_ps(vSY, vSY));
                __m256 vMask = _mm256_cmp_ps(vMagSq, vMagThresh, _CMP_GT_OQ);

                __m256 vInvMag = _mm256_rsqrt_ps(vMagSq);
                __m256 vU = _mm256_loadu_ps(&mU[m]);
                __m256 vV = _mm256_loadu_ps(&mV[m]);
                __m256 vDot = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(vU, vSX), _mm256_mul_ps(vV, vSY)), vInvMag);

                vTotalScore = _mm256_add_ps(vTotalScore, _mm256_and_ps(vMask, vDot));
                vValidCount = _mm256_add_ps(vValidCount, _mm256_and_ps(vMask, vOne));

                // 处理完 16 个点（2个 AVX 块）后检查一次
                if (m == 8 && nPts > 24) {
                    float curVC = _mm256_reduce_add_ps(vValidCount);
                    float curTS = _mm256_reduce_add_ps(vTotalScore);

                    if (curVC < (nPts / 20 * 0.355f)) {
                        passEarlyStage = false;
                        break;
                    }
                }
            }

            if (!passEarlyStage) continue;

            float finalValid = _mm256_reduce_add_ps(vValidCount);
            if (finalValid > nPts * 0.05f) {
                float finalScore = _mm256_reduce_add_ps(vTotalScore) / nPts;
                if (finalScore >= minScore) {
                    allCandidates.push_back({ (float)c, (float)r, 0, 0, finalScore });
                }
            }
        }
    }

    // NMS 处理
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

void downsample2x2_simd(const cv::Mat& src, cv::Mat& dst) {
    int dstRows = src.rows >> 1;
    int dstCols = src.cols >> 1;
    dst.create(dstRows, dstCols, CV_16SC1); // 指定是 16-bit signed

    __m256i ones = _mm256_set1_epi16(1);

    for (int r = 0; r < dstRows; ++r) {
        const short* s1 = src.ptr<short>(2 * r);
        const short* s2 = src.ptr<short>(2 * r + 1);
        short* d = dst.ptr<short>(r);

        int c = 0;
        // 每次处理 16 个输出像素
        for (; c <= dstCols - 16; c += 16) {
            __m256i row1_0 = _mm256_loadu_si256((const __m256i*)(s1 + 2 * c));      // pixel 0-15
            __m256i row1_1 = _mm256_loadu_si256((const __m256i*)(s1 + 2 * c + 16)); // pixel 16-31
            __m256i row2_0 = _mm256_loadu_si256((const __m256i*)(s2 + 2 * c));
            __m256i row2_1 = _mm256_loadu_si256((const __m256i*)(s2 + 2 * c + 16));

            // 列加法
            __m256i vsum0 = _mm256_add_epi16(row1_0, row2_0);
            __m256i vsum1 = _mm256_add_epi16(row1_1, row2_1);

            // 行加法
            __m256i hsum0 = _mm256_madd_epi16(vsum0, ones);
            __m256i hsum1 = _mm256_madd_epi16(vsum1, ones);

            __m256i res32_0 = _mm256_srai_epi32(hsum0, 2); // 2
            __m256i res32_1 = _mm256_srai_epi32(hsum1, 2);

            // 修正顺序
            __m256i packed = _mm256_packs_epi32(res32_0, res32_1);
            __m256i final = _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));

            _mm256_storeu_si256((__m256i*)(d + c), final);
        }

        // 边界处理
        for (; c < dstCols; ++c) {
            d[c] = (short)((s1[2 * c] + s1[2 * c + 1] + s2[2 * c] + s2[2 * c + 1]) >> 2);
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
    std::vector<std::vector<Points>> pyramidModels,
    int numLevels
) {
    if (numLevels < 1) return {};

    for (int i = 1; i < numLevels; ++i) {
        downsample2x2_simd(pyramidGx[i - 1], pyramidGx[i]);
        downsample2x2_simd(pyramidGy[i - 1], pyramidGy[i]);

        //downsample2x2(pyramidGx[i - 1], pyramidGx[i]);
        //downsample2x2(pyramidGy[i - 1], pyramidGy[i]);

        float factor = std::pow(2.0f, (float)i);
        for (const auto& pt0 : pyramidModels[0]) {
            Points lowPt = pt0;
            lowPt.dx /= factor;
            lowPt.dy /= factor;
            pyramidModels[i].push_back(lowPt);
        }
    }

    unsigned int topIdx = numLevels - 1;
     
    // 寻找顶层的粗特征
    std::vector<Points> LayerResults = findAllMatches(
        pyramidGx[topIdx], pyramidGy[topIdx],
        pyramidModels[topIdx],
        0.7f, // 匹配分数
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

            // 5 * 5
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

    for (Points& candidate : results) {
        bool isDuplicate = false; // 是否复制
        for (const Points& confirmed : kept) {

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