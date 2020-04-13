#pragma once

#include <functional>

namespace cppmllib
{
	std::function<double(const double&)> linearRegression(const std::vector<double>& output, const std::vector<double>& input, double& slope, double& bias);
}