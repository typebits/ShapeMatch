#include "shapeMatch.h"

// 高性能水平累加
inline float _mm256_reduce_add_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, lo);
    lo = _mm_add_ss(lo, shuf);
    return _mm_cvtss_f32(lo);
}

std::vector<Points> findAllMatches(const cv::Mat& tgx, const cv::Mat& tgy,
    std::vector<Points> modelPoints, float minScore, float minDist) {

    const int rows = tgx.rows;
    const int cols = tgx.cols;
    const size_t nPts = modelPoints.size();
    std::vector<Points> allCandidates;
    if (nPts == 0) return allCandidates;

    // --- 1. 预处理：Float 转换与平滑 (必须移出主循环) ---
    cv::Mat tgxf, tgyf;
    tgx.convertTo(tgxf, CV_32F);
    tgy.convertTo(tgyf, CV_32F);
    // 可选：在这里做一次 2x2 boxFilter 代替之前的 (p+p)>>1 逻辑

    const float* pBaseX = (const float*)tgxf.data;
    const float* pBaseY = (const float*)tgyf.data;
    const int step = (int)(tgxf.step / sizeof(float));

    // SoA 布局
    size_t alignedPts = (nPts + 7) & ~7;
    std::vector<float> mDX(alignedPts, 0), mDY(alignedPts, 0), mU(alignedPts, 0), mV(alignedPts, 0);
    for (size_t i = 0; i < nPts; ++i) {
        mDX[i] = modelPoints[i].dx; mDY[i] = modelPoints[i].dy;
        mU[i] = modelPoints[i].u;   mV[i] = modelPoints[i].v;
    }

    const __m256 vMagThresh = _mm256_set1_ps(225.0f);
    const __m256 vOne = _mm256_set1_ps(1.0f);
    const __m256i vStep = _mm256_set1_epi32(step);

    // --- 2. 核心并行循环 ---
#pragma omp parallel
    {
        std::vector<Points> localCandidates;
#pragma omp for
        for (int r = 0; r < rows - 1; ++r) {
            const __m256 vR = _mm256_set1_ps((float)r);

            // 一次处理 4 个水平像素块
            for (int c = 0; c < cols - 4; c += 4) {
                __m256 vScore0 = _mm256_setzero_ps(), vScore1 = _mm256_setzero_ps();
                __m256 vScore2 = _mm256_setzero_ps(), vScore3 = _mm256_setzero_ps();
                __m256 vCount0 = _mm256_setzero_ps(), vCount1 = _mm256_setzero_ps();
                __m256 vCount2 = _mm256_setzero_ps(), vCount3 = _mm256_setzero_ps();

                const __m256 vC0 = _mm256_set1_ps((float)c);
                const __m256 vC1 = _mm256_set1_ps((float)(c + 1));
                const __m256 vC2 = _mm256_set1_ps((float)(c + 2));
                const __m256 vC3 = _mm256_set1_ps((float)(c + 3));

                bool active[4] = { true, true, true, true };

                for (size_t m = 0; m < alignedPts; m += 8) {
                    __m256 vDX = _mm256_loadu_ps(&mDX[m]);
                    __m256 vDY = _mm256_loadu_ps(&mDY[m]);
                    __m256 vU = _mm256_loadu_ps(&mU[m]);
                    __m256 vV = _mm256_loadu_ps(&mV[m]);

                    __m256i vYIdx = _mm256_cvttps_epi32(_mm256_add_ps(vR, vDY));
                    __m256i vYPart = _mm256_mullo_epi32(vYIdx, vStep);

                    // 计算 4 个位置的 Gather 偏移
                    __m256i vOff0 = _mm256_add_epi32(vYPart, _mm256_cvttps_epi32(_mm256_add_ps(vC0, vDX)));
                    __m256i vOff1 = _mm256_add_epi32(vYPart, _mm256_cvttps_epi32(_mm256_add_ps(vC1, vDX)));
                    __m256i vOff2 = _mm256_add_epi32(vYPart, _mm256_cvttps_epi32(_mm256_add_ps(vC2, vDX)));
                    __m256i vOff3 = _mm256_add_epi32(vYPart, _mm256_cvttps_epi32(_mm256_add_ps(vC3, vDX)));

                    // 批量发射 Gather (流水线并行核心)
                    __m256 vSX0 = _mm256_i32gather_ps(pBaseX, vOff0, 4);
                    __m256 vSX1 = _mm256_i32gather_ps(pBaseX, vOff1, 4);
                    __m256 vSX2 = _mm256_i32gather_ps(pBaseX, vOff2, 4);
                    __m256 vSX3 = _mm256_i32gather_ps(pBaseX, vOff3, 4);

                    __m256 vSY0 = _mm256_i32gather_ps(pBaseY, vOff0, 4);
                    __m256 vSY1 = _mm256_i32gather_ps(pBaseY, vOff1, 4);
                    __m256 vSY2 = _mm256_i32gather_ps(pBaseY, vOff2, 4);
                    __m256 vSY3 = _mm256_i32gather_ps(pBaseY, vOff3, 4);

                    // 计算逻辑函数 (Lambda 简化代码阅读)
                    auto computeBlock = [&](__m256 sx, __m256 sy, __m256& score, __m256& count) {
                        __m256 magSq = _mm256_add_ps(_mm256_mul_ps(sx, sx), _mm256_mul_ps(sy, sy));
                        __m256 mask = _mm256_cmp_ps(magSq, vMagThresh, _CMP_GT_OQ);
                        __m256 dot = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(vU, sx), _mm256_mul_ps(vV, sy)), _mm256_rsqrt_ps(magSq));
                        score = _mm256_add_ps(score, _mm256_and_ps(mask, dot));
                        count = _mm256_add_ps(count, _mm256_and_ps(mask, vOne));
                        };

                    if (active[0]) computeBlock(vSX0, vSY0, vScore0, vCount0);
                    if (active[1]) computeBlock(vSX1, vSY1, vScore1, vCount1);
                    if (active[2]) computeBlock(vSX2, vSY2, vScore2, vCount2);
                    if (active[3]) computeBlock(vSX3, vSY3, vScore3, vCount3);

                    // 早期退出检查 (针对每个 Block 独立判断)
                    if (m == 16 && nPts > 400) {
                        if (active[0] && _mm256_reduce_add_ps(vCount0) < (nPts * 0.1f)) active[0] = false;
                        if (active[1] && _mm256_reduce_add_ps(vCount1) < (nPts * 0.1f)) active[1] = false;
                        if (active[2] && _mm256_reduce_add_ps(vCount2) < (nPts * 0.1f)) active[2] = false;
                        if (active[3] && _mm256_reduce_add_ps(vCount3) < (nPts * 0.1f)) active[3] = false;
                        if (!(active[0] || active[1] || active[2] || active[3])) break;
                    }
                }

                // 结果处理
                auto finalize = [&](int colOffset, __m256 count, __m256 score) {
                    float fCount = _mm256_reduce_add_ps(count);
                    if (fCount > nPts * 0.5f) {
                        float fScore = _mm256_reduce_add_ps(score) / nPts;
                        if (fScore >= minScore) localCandidates.push_back({ (float)(c + colOffset), (float)r, 0, 0, fScore });
                    }
                    };

                if (active[0]) finalize(0, vCount0, vScore0);
                if (active[1]) finalize(1, vCount1, vScore1);
                if (active[2]) finalize(2, vCount2, vScore2);
                if (active[3]) finalize(3, vCount3, vScore3);
            }
        }
#pragma omp critical
        allCandidates.insert(allCandidates.end(), localCandidates.begin(), localCandidates.end());
    }

    if (allCandidates.size() > 1) applyNMS(allCandidates, minDist);
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
        0.55f, // 匹配分数
        4.0f // 距离
    );

    for (int level = topIdx - 1; level >= 0; --level) {
        std::vector<Points> nextResults;
        nextResults.reserve(LayerResults.size() + LayerResults.size());

        const auto& curGx = pyramidGx[level];
        const auto& curGy = pyramidGy[level];
        const auto& curModel = pyramidModels[level];
        const int rows = curGx.rows;
        const int cols = curGx.cols;

        float currentThreshold = 0.55f;

        for (const auto& prevRes : LayerResults) {
            const int cx = static_cast<int>(prevRes.dx + prevRes.dx + 0.5f);
            const int cy = static_cast<int>(prevRes.dy + prevRes.dy + 0.5f);

            // 5 * 5
            const int rStart = std::max(0, cy - 2);
            const int rEnd = std::min(rows - 1, cy + 2);
            const int cStart = std::max(0, cx - 2);
            const int cEnd = std::min(cols - 1, cx + 2);

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