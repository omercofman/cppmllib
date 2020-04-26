#include <iostream>
#include <vector>
#include <random>
#include <tuple>
#include <string>

#include "regression.h"
#include "utilities.h"


using std::vector;
using std::tuple;
using std::string;
using std::cout;
using std::endl;

void testRegression();

int main()
{
	testRegression();
	return 0;
}

void testLinearRegression() {
	vector<vector<double>> x{ { 1, 2, 4, 3, 5 } };
	vector<double> y{ 1, 3, 3, 2, 5 };

	vector<tuple<string, cppmllib::regression, double, size_t>> linearFunc{
		{ "cppmllib::linearRegression",			cppmllib::linearRegression,			-1,		-1 },
		{ "cppmllib::linearGradientDescent",	cppmllib::linearGradientDescent,	0.01,	20 },
	};

	enum {
		NAME,
		FUNC,
		ALPH,
		EPOC
	};

	for (auto obj : linearFunc) {
		cout << std::get<NAME>(obj) << endl;

		vector<double> coeff(1 + x.size(), 0.0);
		auto estfunc = std::get<FUNC>(obj)(y, x, coeff, std::get<ALPH>(obj), std::get<EPOC>(obj));

		for (auto i{ 0U }; i < coeff.size(); ++i) {
			cout << "coeff[" << i << "]: " << coeff[i] << endl;
		}

		for (auto i{ 0U }; i < x[0].size(); ++i) {
			auto predicted = estfunc({ x[0][i] });
			cout << "Predicted: " << predicted << "\tActuall: " << y[i] << "\tError: " << y[i] - predicted << endl;
		}

		cout << "RMSE: " << cppmllib::rootMeanSquareError(y, x[0], estfunc) << endl << endl;
	}
}

void testLogisticRegression() {
	vector<double> x1{ 2.7810836, 1.465489372, 3.396561688, 1.38807019, 3.06407232, 7.627531214, 5.332441248, 6.922596716, 8.675418651, 7.673756466 };
	vector<double> x2{ 2.550537003, 2.362125076, 4.400293529, 1.850220317, 3.005305973, 2.759262235, 2.088626775, 1.77106367, -0.242068655, 3.508563011 };

	vector<vector<double>> x{ x1, x2 };
	vector<double> y{ 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 };

	auto alpha{ 0.3 };
	auto epochs{ 10U };

	vector<double> coeff(1 + x.size(), 0.0);
	auto estfunc = cppmllib::logisticRegression(y, x, coeff, alpha, epochs);

	cout << "cppmllib::logisticRegression" << endl;

	for (auto i{ 0U }; i < coeff.size(); ++i) {
		cout << "coeff[" << i << "]: " << coeff[i] << endl;
	}

	auto hit{ 0U };

	for (auto i{ 0U }; i < x1.size(); ++i) {
		auto predicted = estfunc({ x1[i], x2[i] }) >= 0.5;
		if (predicted == y[i]) {
			++hit;
		}

		cout << "Predicted: " << predicted << "\tActuall: " << y[i] << "\tError: " << y[i] - predicted << endl;
	}

	cout << "Accuracy estimation: " << 100.0 * hit / x1.size() << "%" << endl << endl;
}

void testRegression() {
	testLinearRegression();
	testLogisticRegression();
}