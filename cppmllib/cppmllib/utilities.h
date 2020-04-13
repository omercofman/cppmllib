#pragma once

#include <vector>
#include <functional>

#define THROW_IF_ZERO(_size_) ((0 == (_size_)) ? throw "Argument can't be 0" : 0)
#define THROW_IF_NOT_EQUAL(_left_, _right_) (((_left_) != (_right_)) ? throw "Arguments Must match" : 0)

namespace cppmllib
{
	double average(const std::vector<double>::const_iterator& cbegin, const std::vector<double>::const_iterator& cend);
	double rootMeanSquareError(const std::vector<double>& output, const std::vector<double>& input,
		const std::function<double(const double&)>& estfunc);
}