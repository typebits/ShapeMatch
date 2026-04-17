#pragma once
#include "shapeMatch.h"

std::vector<Points> extractModelPoints(const cv::Mat& src);

std::vector<Points> rotateModel(const std::vector<Points>& srcModel, float angleDeg);