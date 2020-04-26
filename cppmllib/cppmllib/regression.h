#pragma once

#include <functional>
#include <vector>

using std::function;
using std::vector;

namespace cppmllib
{
	typedef function<function<double(const vector<double>&)>(const vector<double>&, const vector<vector<double>>&, vector<double>&, double, size_t)> regression;
	
	function<double(const vector<double>&)> linearRegression(const vector<double>& codomain, const vector<vector<double>>& domains, vector<double>& coeff, double alpha = -1, size_t epochs = -1);
	function<double(const vector<double>&)> linearGradientDescent(const vector<double>& codomain, const vector<vector<double>>& domains, vector<double>& coeff, double alpha = 0.01, size_t epochs = 20);
	function<double(const vector<double>&)> logisticRegression(const vector<double>& codomain, const vector<vector<double>>& domains, vector<double>& coeff, double alpha = 0.3, size_t epochs = 6);
}