#include <iostream>
#include <vector>

#include "LinearRegression.h"
#include "utilities.h"

void testLinearRegression();

int main()
{
	testLinearRegression();
	return (0);
}

void testLinerRegressionFunctions(const std::vector<double>& y, const std::vector<double>& x, const cppmllib::lrType& lr)
{
	double slope;
	double bias;
	auto estfunc = lr(y, x, slope, bias);
	std::cout << "slope: " << slope << "\tbias: " << bias << std::endl;

	for (size_t i{ 0 }; i < x.size(); ++i)
	{
		auto predicted = estfunc(x.at(i));
		std::cout << "Predicted: " << predicted << "\tActuall: " << y.at(i) << "\tError: " << y.at(i) - predicted << std::endl;
	}
	std::cout << "RMSE Score: " << cppmllib::rootMeanSquareError(y, x, estfunc) << std::endl;
}

void testLinearRegression()
{
	std::vector<double> x{ 1,2,4,3,5 };
	std::vector<double> y{ 1,3,3,2,5 };
	std::vector<cppmllib::lrType> lrFunctions;

	lrFunctions.push_back(cppmllib::linearRegression);
	lrFunctions.push_back(cppmllib::gradientDescent);

	for (auto func : lrFunctions)
	{
		testLinerRegressionFunctions(y, x, func);
		std::cout << std::endl;
	}
}