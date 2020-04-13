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

void testLinearRegression()
{
	std::vector<double> x{ 1,2,4,3,5 };
	std::vector<double> y{ 1,3,3,2,5 };
	double slope, bias;
	auto estfunc = cppmllib::linearRegression(y, x, slope, bias);
	std::cout << "slope: " << slope << "\tbias: " << bias << std::endl;

	for (size_t i{ 0 }; i < x.size(); ++i)
	{
		auto predicted = estfunc(x.at(i));
		std::cout << "Predicted: " << predicted << "\tActuall: " << y.at(i) << "\tError: " << y.at(i) - predicted << std::endl;
	}
	std::cout << "RMSE Score: " << cppmllib::rootMeanSquareError(y, x, estfunc) << std::endl;
}