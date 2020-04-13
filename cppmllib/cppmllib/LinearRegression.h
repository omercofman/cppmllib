#pragma once

#include <functional>

namespace cppmllib
{
	typedef std::function<std::function<double(const double&)>(const std::vector<double>&, const std::vector<double>&, double&, double&)> lrType;
	std::function<double(const double&)> linearRegression(const std::vector<double>& output, const std::vector<double>& input, double& slope, double& bias);
	std::function<double(const double&)> gradientDescent(const std::vector<double>& output, const std::vector<double>& input, double& slope, double& bias);
}