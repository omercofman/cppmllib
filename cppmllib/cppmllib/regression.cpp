
#include "regression.h"
#include "utilities.h"

#include <math.h>

#include <vector>
#include <iterator> 
#include <algorithm>
#include <numeric>

using std::function;
using std::vector;

namespace cppmllib
{

	/*
	*	Sanity check
	*	True for all regression type functions in this file
	*/
	void genericCheckup(const vector<double>& codomain, const vector<vector<double>>& domains, const vector<double>& coeffs, double alpha, size_t epochs) {
		THROW_IF_ZERO(codomain.size());
		
		for (auto i{ 0U }; i < domains.size(); ++i) {
			THROW_IF_NOT_EQUAL(codomain.size(), domains[i].size());
		}
		
		THROW_IF_NOT_EQUAL(coeffs.size(), 1 + domains.size());
		
		THROW_IF_ZERO(alpha);
		THROW_IF_ZERO(epochs);
	}

	/*
	*    y = b0 + b1*x0 + b2*x1 ... b(n-1)*x(n-2)
	*    b0 is the bias
	*    bi coefficients are slopes
	*/
	function<double(const vector<double>&)> hyperplaneFormula(const vector<double>& coeff) {
		auto func = [coeff](const vector<double>& dim) {
			double result{ coeff[0] };

			THROW_IF_NOT_EQUAL(coeff.size(), 1 + dim.size());

			for (auto i{ 0U }; i < dim.size(); ++i) {
				result += coeff[1 + i] * dim[i];
			}
			return result;
		};

		return func;
	}

	/*
	*	N-Dimentions Liner Regression f:R^N->R
	*
	*	Coefficient calculation:
	*	Coefficient i=1...(N-1) = SUM j=0...(N-1)[(xij - mean(xi)) * (yi - mean(y))] / SUM j=0...(N-1)[(xij - mean(xi))^2]
	*	Coefficient0 = mean(y) - SUM j=1...(N-1)[ci * x(i-1)]
	*
	*	Estimated function calculation:
	*   f(values) -> c0 + SUM i=1...(N-1)[ci * values(i-1)]
	*
	*	Input:
	*	const vector<double>&			codomain	- Sulotions of the function
	*   const vector<vector<double>>&	domains		- The dimentios of the function 
	*	vector<double>&					coeff		- Coefficients initialized to first guess
	*	dobule							alpha		- not used (deafult -1)
	*	size_t							epochs		- not used (deafult -1)
	*
	*	Output:
	*	function<double(const vector<double>&)>		- Estimated function
	*/
	function<double(const vector<double>&)> linearRegression(const vector<double>& codomain, const vector<vector<double>>& domains, vector<double>& coeff, double alpha, size_t epochs) {
		genericCheckup(codomain, domains, coeff, -1, -1);

		auto codomainAverage{ cppmllib::average(codomain) };
		vector<double> domainAverage;
		
		for (auto i{ 0U }; i < domains.size(); ++i) {
			auto currDomain{ domains[i] };
			auto currDomainAverage{ cppmllib::average(currDomain) };
			
			domainAverage.push_back(currDomainAverage);

			auto sumMultiResiduals{ 0.0 };
			auto sumSquareResiduals{ 0.0 };
			
			for (auto j{ 0U }; j < currDomain.size(); ++j) {
				auto domainResidual{ currDomain[j] - currDomainAverage };
				sumMultiResiduals += domainResidual * (codomain[j] - codomainAverage);	// Preparing the numerator
				sumSquareResiduals += domainResidual * domainResidual;					// Prepering the denominator 
			}

			THROW_IF_ZERO(sumSquareResiduals);
			coeff[1 + i] = sumMultiResiduals / sumSquareResiduals;
		}
		
		coeff[0] = codomainAverage;

		for (auto i{ 1U }; i < coeff.size(); ++i) {
			coeff[0] -= coeff[i] * domainAverage[i - 1];
		}

		return hyperplaneFormula(coeff);
	}

	/*
	*	N-Dimentions Logistic Regression f:R^N->R(0...1)
	*
	*	Coefficient calculation:
	*	Coefficient i=1...(N-1) = ci + alpha * (ci - est) * est * (1 - est) * xi
	*   Coefficient0 = c0 + alpha * (c0 - est) * est * (1 - est)
	*
	*	Estimated function calculation:
	*   f(values) -> 1 / (1 + e^-(c0 + SUM i=1...(N-1)[ci * values(i-1)]))
	*
	*	Input:
	*	const vector<double>&			codomain	- Sulotions of the function
	*   const vector<vector<double>>&	domains		- The dimentios of the function
	*	vector<double>&					coeff		- Coefficients initialized to first guess
	*	dobule							alpha		- Change rate
	*	size_t							epochs		- Times to run through the dataset
	*
	*	Output:
	*	function<double(const vector<double>&)>		- Estimated function
	*/
	function<double(const vector<double>&)> logisticRegression(const vector<double>& codomain, const vector<vector<double>>& domains, vector<double>& coeff, double alpha, size_t epochs) {
		genericCheckup(codomain, domains, coeff, alpha, epochs);

		for (auto toLoop{ 0U }; toLoop < epochs; ++toLoop) {
			for (auto i{ 0U }; i < codomain.size(); ++i) {
				auto localPrediction = hyperplaneFormula(coeff); // Prepare the prediction function with the current coefficients
				vector<double> dim;

				for (size_t j{ 0 }; j < domains.size(); ++j) {
					dim.push_back(domains[j][i]);
				}

				// Find current estimation
				auto estimation = 1 / (1 + exp(-localPrediction(dim)));
				
				// Update coefficients
				coeff[0] = coeff[0] + alpha * (codomain[0] - estimation) * estimation * (1 - estimation);
				for (auto j{ 1U }; j < coeff.size(); ++j) {					
					coeff[j] = coeff[j] + alpha * (codomain[i] - estimation) * estimation * (1 - estimation) * domains[j - 1][i];
				}
			}
		}

		auto prediction = hyperplaneFormula(coeff);
		auto logisticEstimation = [=](const vector<double>& dim) {
			return (1 / (1 + exp(-prediction(dim))));
		};
		return logisticEstimation;
	}

	/*
	*	N-Dimentions Gradient Descent f:R^N->R
	*
	*	Coefficient calculation:
	*	Coefficient 
	*   Coefficient0 
	*
	*	Estimated function calculation:
	*   f(values) -> 
	*
	*	Input:
	*	const vector<double>&			codomain	- Sulotions of the function
	*   const vector<vector<double>>&	domains		- The dimentios of the function
	*	vector<double>&					coeff		- Coefficients initialized to first guess
	*	dobule							alpha		- Change rate
	*	size_t							epochs		- Times to run through the dataset
	*
	*	Output:
	*	function<double(const vector<double>&)>		- Estimated function
	*/
	function<double(const vector<double>&)> gradientDescent(const vector<double>& codomain, const vector<vector<double>>& domains, vector<double>& coeff, double alpha, size_t epochs) {
		genericCheckup(codomain, domains, coeff, alpha, epochs);
		
		for (auto toLoop{ 0U }; toLoop < epochs; ++toLoop) {
			for (auto i{ 0U }; i < domains.size(); ++i) {
				auto prediction = hyperplaneFormula(coeff);
				auto error = prediction(domains[i]) - codomain[i];
				
				coeff[0] -= alpha * error;

				for (auto j{ 1U }; j < coeff.size(); ++j) {
					coeff[j] -= alpha * error * domains[i][j - 1];
				}
			}			
		}

		return hyperplaneFormula(coeff);
	}
}

