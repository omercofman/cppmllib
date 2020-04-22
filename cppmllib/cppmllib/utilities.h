#pragma once

#include <vector>
#include <functional>

#define THROW_IF_ZERO(_size_) ((0 == (_size_)) ? throw "Size can't be 0" : (0))
#define THROW_IF_NOT_EQUAL(_left_, _right_) (((_left_) != (_right_)) ? throw "Sizes Must match" : (0))

using std::vector;
using std::function;

namespace cppmllib
{
	double average(const vector<double>::const_iterator& cbegin, const vector<double>::const_iterator& cend);
	double average(const vector<double>& vec);
	double rootMeanSquareError(const vector<double>& codomain, const vector<double>& dim, const function<double(const vector<double>&)>& estfunc);
}