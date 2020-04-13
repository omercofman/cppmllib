
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
	*	Slope = SUM[i=0...N-1]((Xi - MEAN(X))*(Yi - MEAN(Y)) / SUM[i=0...N-1]((Xi - MEAN(X))**2) 
	*   Bias = MEAN(Y) - Slope * MEAN(X)
	*
	*   Returns:
	*   std::function<double (const double&)> Estimation (prediction) function defined as {bias + slope * x}
	*/
	std::function<double (const double&)> linearRegression(const std::vector<double>& output, const std::vector<double>& input, double& slope, double& bias)
	{
		THROW_IF_ZERO(output.size());
		THROW_IF_NOT_EQUAL(output.size(), input.size());

		double avg_output = cppmllib::average(output.cbegin(), output.cend());
		double avg_input = cppmllib::average(input.cbegin(), input.cend());
		double sum_multi_of_residuals{ 0 };
		double sum_square_input_residuals{ 0 };

		for (size_t i{ 0 }; i < output.size(); ++i)
		{
			auto curr_residual_input = input.at(i) - avg_input;						// Xi - MEAN(X)
			auto curr_residual_output = output.at(i) - avg_output;					// Yi - MEAN(Y)
			sum_multi_of_residuals += curr_residual_input * curr_residual_output;	// SUM[i=0...N-1]((Xi - MEAN(X))*(Yi - MEAN(Y))
			sum_square_input_residuals += curr_residual_input * curr_residual_input;// SUM[i=0...N-1]((Xi - MEAN(X))**2)
		}

		slope = sum_multi_of_residuals / sum_square_input_residuals;
		bias = avg_output - (slope * avg_input);

		auto estfunc = [=](const double& x) -> double
		{
			return (bias + (slope * x));
		};
		return estfunc;
	}

	/*
	*   Gradient Descent
	*
	*	Input arguments:
	*	const std::vector<double>& output - Vector of outputs of an unknown function (Y axis)
	*	const std::vector<double>& input - Vector of outputs of an unknown function (Y axis)
	*
	*   Output arguments:
	*   double& slope - The slope of the estimated function
	*   double& bias - The bias of the estimated function
	*
	*   ***under work***
	*   Optional arguments:
	*   const double alpha - Rate of change of slope and bias based on the error of estimation
	*   const size_t epochs - Number of iterations throght the database (output and input vectors)
   	*
	*   The output arguments are calculated in this fasion:
	*	Starting with a guess of (0,0) for (slope,bias) arguments, GD find the error between the temporary
	*	estimated (predicted) function and the output vector.
	*	Slope and bias are amendment to a new value based on thier old value and the rate of change (alpha).
	*
	*   Returns:
	*   std::function<double (const double&)> Estimation (prediction) function defined as {bias + slope * x}
	*/
	std::function<double(const double&)> gradientDescent(const std::vector<double>& output, const std::vector<double>& input, double& slope, double& bias,
		const double alpha, const size_t epochs)
	{
		THROW_IF_ZERO(output.size());
		THROW_IF_NOT_EQUAL(output.size(), input.size());

		// Initial guess
		slope = 0;
		bias = 0;

		// Prediction function
		auto prediction = [](const double& bias, const double& slope, const double& x) -> double
		{
			return (bias + (slope * x));
		};

		for (size_t j{ 0 }; j < epochs; ++j)
		{
			for (size_t i{ 0 }; i < output.size(); ++i)
			{
				auto error = prediction(bias, slope, input.at(i)) - output.at(i);
				bias -= error * alpha;
				slope -= error * alpha * input.at(i);
			}
		}

		auto estfunc = [=](const double& x) -> double
		{
			return (bias + (slope * x));
		};
		return estfunc;
	}

	std::function<double(const double&)> gradientDescent(const std::vector<double>& output, const std::vector<double>& input, double& slope, double& bias)
	{
		/*
		*	I would like all the function to have the same signature and
		*	in the name time to have optional/default arguments.
		*	I didn't found a way to do it at the moment
		*/
		return gradientDescent(output, input, slope, bias, 0.01, 4);
	}
}

















