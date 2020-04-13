
#include <vector>
#include <iterator> 
#include <algorithm>
#include <numeric>

#include "LinearRegression.h"
#include "utilities.h"

namespace cppmllib
{
	/*
	*   Simple Linear Regression
	*
	*	Input arguments:
	*	const std::vector<double>& output - Vector of outputs of an unknown function (Y axis)
	*	const std::vector<double>& input - Vector of outputs of an unknown function (Y axis)
	*
	*   Output arguments:
	*   double& slope - The slope of the estimated function
	*   double& bias - The bias of the estimated function
	*   
	*   The output arguments are calculated in this fasion:
	*	Slope = SUM[i=0...N]((Xi - MEAN(X))*(Yi - MEAN(Y)) / SUM[i=0...N]((Xi - MEAN(X))**2) 
	*   Bias = MEAN(Y) - Slope * MEAN(X)
	*
	*   Returns:
	*   std::function<double (const double&)> Estimation (prediction) function defined as {bias + slope * x}
	*/
	std::function<double (const double&)> linearRegression(const std::vector<double>& output, const std::vector<double>& input, double& slope, double& bias)
	{
		THROW_IF_NOT_EQUAL(output.size(), input.size());

		double avg_output = cppmllib::average(output.cbegin(), output.cend());
		double avg_input = cppmllib::average(input.cbegin(), input.cend());
		double sum_multi_of_residuals{ 0 };
		double sum_square_input_residuals{ 0 };

		for (size_t i{ 0 }; i < output.size(); ++i)
		{
			auto curr_residual_input = input.at(i) - avg_input;
			auto curr_residual_output = output.at(i) - avg_output;
			sum_multi_of_residuals += curr_residual_input * curr_residual_output;
			sum_square_input_residuals += curr_residual_input * curr_residual_input;
		}

		slope = sum_multi_of_residuals / sum_square_input_residuals;
		bias = avg_output - (slope * avg_input);

		auto estfunc = [bias, slope](const double& x) -> double
		{
			return (bias + (slope * x)); 
		};
		return estfunc;
	}
}

