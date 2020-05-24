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
		if (predicted == static_cast<bool>(y[i])) {
			++hit;
		}

		cout << "Predicted: " << predicted << "\tActuall: " << y[i] << "\tError: " << y[i] - predicted << endl;
	}

	cout << "Accuracy estimation: " << 100.0 * hit / x1.size() << "%" << endl << endl;
}

void testLDA() {
	vector<double> x1{ 4.667797637, 5.509198779, 4.702791608, 5.956706641, 5.738622413, 5.027283325, 4.805434058, 4.425689143, 5.009368635, 5.116718815, 6.370917709, 2.895041947, 4.666842365, 5.602154638, 4.902797978, 5.032652964, 4.083972925, 4.875524106, 4.732801047, 5.385993407, 20.74393514, 21.41752855, 20.57924186, 20.7386947 , 19.44605384, 18.36360265, 19.90363232, 19.10870851, 18.18787593, 19.71767611, 19.09629027, 20.52741312, 20.63205608, 19.86218119, 21.34670569, 20.333906, 21.02714855, 18.27536089, 21.77371156, 20.65953546 };

	vector<vector<double>> x{ x1 };
	vector<double> y(x1.size(), 0);
	 
	for (auto i{ (y.size() >> 1) }; i < y.size(); ++i) {
		y[i] = 1;
	}

	auto alpha{ -1.0 };
	auto epochs{ 0U };

	vector<double> coeff(1 + x.size(), -1.0);
	auto estfunc = cppmllib::linearDiscriminantAnalysis(y, x, coeff, alpha, epochs);

	cout << "cppmllib::linearDiscriminantAnalysis" << endl;

	for (auto i{ 0U }; i < x1.size(); ++i) {
		auto predicted = estfunc({ x1[i] });
		cout << "Predicted: " << predicted << "\tActuall: " << y[i] << "\tError: " << y[i] - predicted << endl;
	}
}

void testRegression() {
	testLinearRegression();
	testLogisticRegression();
	testLDA();
}