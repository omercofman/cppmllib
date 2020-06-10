#include <iostream>
#include <vector>
#include <random>
#include <tuple>
#include <string>
#include <functional>

#include "linearAlgorithms.h"
#include "nonlinearAlgorithms.h"
#include "utilities.h"


using std::vector;
using std::tuple;
using std::string;
using std::function;

using std::cout;
using std::endl;

void testLinearAlgorithms();

int main()
{
	testLinearAlgorithms();
	return 0;
}

void testLinearAlgorithms() {
	vector<vector<double>> *x;
	vector<double> *y;
	function<double(const vector<double>&)> estfunc;
	vector<double> *coeff;
	
	auto eval_regression = [&]() {
		for (auto i{ 0U }; i < coeff->size(); ++i) {
			cout << "coeff[" << i << "]: " << coeff->at(i) << endl;
		}

		for (auto i{ 0U }; i < x->at(0).size(); ++i) {
			auto predicted = estfunc({ x->at(0).at(i) }); 
			cout << "Predicted: " << predicted << "\tActuall: " << y->at(i) << "\tError: " << y->at(i) - predicted << endl;
		}

		cout << "RMSE: " << cppmllib::rootMeanSquareError(*y, x->at(0), estfunc) << endl << endl;
	};

	auto eval_binary = [&]() {
		for (auto i{ 0U }; i < coeff->size(); ++i) {
			cout << "coeff[" << i << "]: " << coeff->at(i) << endl;
		}

		auto hit{ 0U };

		for (auto i{ 0U }; i < x->at(0).size(); ++i) {
			auto predicted = estfunc({ x->at(0)[i], x->at(1)[i] }) >= 0.5;
			if (predicted == static_cast<bool>(y->at(i))) {
				++hit;
			}

			cout << "Predicted: " << predicted << "\tActuall: " << y->at(i) << "\tError: " << y->at(i) - predicted << endl;
		}

		cout << "Accuracy estimation: " << 100.0 * hit / x->at(0).size() << "%" << endl << endl;
	};
	
	auto eval_lda = [&]() {
		for (auto i{ 0U }; i < x->at(0).size(); ++i) {
			auto predicted = estfunc({ x->at(0)[i] });
			cout << "Predicted: " << predicted << "\tActuall: " << y->at(i) << "\tError: " << y->at(i) - predicted << endl;
		}
	};

	vector<tuple<string, cppmllib::linearAlgorithm, vector<double>, vector<vector<double>>, double, size_t, function<void(void)>>> database{
		//Function name                         function                         y                  x                           alpha   epoch    evaluation function
		{ "cppmllib::linearRegression",         cppmllib::linearRegression,      { 1, 3, 3, 2, 5 }, {{ 1, 2, 4, 3, 5 }},        -1.0,   0U - 1,  eval_regression },
		{ "cppmllib::linearGradientDescent",    cppmllib::linearGradientDescent, { 1, 3, 3, 2, 5 }, {{ 1, 2, 4, 3, 5 }},        0.01,   20U,     eval_regression },
		{ "cppmllib::logisticRegression",       cppmllib::logisticRegression,    { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 },
			// x
			{{ 2.7810836, 1.465489372, 3.396561688, 1.38807019, 3.06407232, 7.627531214, 5.332441248, 6.922596716, 8.675418651, 7.673756466 },     //x0
			{ 2.550537003, 2.362125076, 4.400293529, 1.850220317, 3.005305973, 2.759262235, 2.088626775, 1.77106367, -0.242068655, 3.508563011 }}, //x1
			//alpha	epoch evaluation function
			0.3,	10U,	eval_binary
		},
		//Function name                             function
		{ "cppmllib::linearDiscriminantAnalysis",   cppmllib::linearDiscriminantAnalysis,
			//y
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			//x
			{{ 4.667797637, 5.509198779, 4.702791608, 5.956706641, 5.738622413, 5.027283325, 4.805434058, 4.425689143, 5.009368635, 5.116718815, 6.370917709, 2.895041947,
			4.666842365, 5.602154638, 4.902797978, 5.032652964, 4.083972925, 4.875524106, 4.732801047, 5.385993407, 20.74393514, 21.41752855, 20.57924186, 20.7386947 ,
			19.44605384, 18.36360265, 19.90363232, 19.10870851, 18.18787593, 19.71767611, 19.09629027, 20.52741312, 20.63205608, 19.86218119, 21.34670569, 20.333906, 21.02714855,
			18.27536089, 21.77371156, 20.65953546 }},
			//alpha epoch	evaluation function
			-1.0,   0U - 1, eval_lda
		},
	};

	enum {
		NAME,
		FUNC,
		Y,
		X,
		ALPH,
		EPOC,
		EVAL
	};

	for (auto tpl : database) {
		cout << std::get<NAME>(tpl) << endl;

		x = &std::get<X>(tpl);
		y = &std::get<Y>(tpl);

		coeff = new vector<double>(1 + x->size(), 0.0);
		estfunc = std::get<FUNC>(tpl)(*y, *x, *coeff, std::get<ALPH>(tpl), std::get<EPOC>(tpl));

		std::get<EVAL>(tpl)();	
	}
}
