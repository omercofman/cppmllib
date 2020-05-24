
#include "regression.h"
#include "utilities.h"

#include <math.h>
#include <float.h>

#include <vector>
#include <iterator> 
#include <algorithm>
#include <numeric>
#include <map>

using std::function;
using std::vector;
using std::map;

namespace cppmllib
{

	/*
	*	Sanity check
	*	True for all regression type functions in this file
	*/
	void genericCheckup(const vector<double>& codomain, const vector<vector<double>>& domains, const vector<double>& coeffs, const double& alpha, const size_t& epochs) {
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
	*    bi is the slope of dimension number i
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
	*	vector<double>&					coeff		- Coefficients initialized to first guess (weights)
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
	*	N-Dimentions Gradient Descent f:R^N->R
	*
	*	Coefficient calculation:
	*	Coefficient i=1...(N-1) = ci - alpha * error * xij
	*   Coefficient0 = c0 - alpha * error
	*
	*	Estimated function calculation:
	*   f(values) -> c0 + SUM i=1...(N-1)[ci * values(i-1)]
	*
	*	Input:
	*	const vector<double>&			codomain	- Sulotions of the function
	*   const vector<vector<double>>&	domains		- The dimentios of the function
	*	vector<double>&					coeff		- Coefficients initialized to first guess (weights)
	*	dobule							alpha		- Change rate
	*	size_t							epochs		- Times to run through the dataset
	*
	*	Output:
	*	function<double(const vector<double>&)>		- Estimated function
	*/
	function<double(const vector<double>&)> linearGradientDescent(const vector<double>& codomain, const vector<vector<double>>& domains, vector<double>& coeff, double alpha, size_t epochs) {
		genericCheckup(codomain, domains, coeff, alpha, epochs);

		for (auto toLoop{ 0U }; toLoop < epochs; ++toLoop) {
			for (auto i{ 0U }; i < codomain.size(); ++i) {
				auto localPrediction = hyperplaneFormula(coeff);
				vector<double> dim;

				for (auto j{ 0U }; j < domains.size(); ++j) {
					dim.push_back(domains[j][i]);
				}

				// Find current error of prediction
				auto error = localPrediction(dim) - codomain[i];

				// Update coefficients
				coeff[0] -= alpha * error;
				for (auto j{ 1U }; j < coeff.size(); ++j) {
					coeff[j] -= alpha * error * dim[j - 1];
				}
			}
		}

		return hyperplaneFormula(coeff);
	}

	/*
	*	N-Dimentions Logistic Regression f:R^N->R (Real values between (0,1))
	*
	*	Coefficient calculation:
	*	Coefficient i=1...(N-1) = ci + alpha * (ci - est) * est * (1 - est) * xi
	*   Coefficient0 = c0 + alpha * (c0 - est) * est * (1 - est)
	*
	*	Estimated function calculation:
	*   f(values) -> 1 / (1 + e^-(c0 + SUM i=1...(N-1)[ci * values(i-1)]))
	*
	*	Input:
	*	const vector<double>&			codomain	- Sulotions of the function (classes)
	*   const vector<vector<double>>&	domains		- The dimentios of the function
	*	vector<double>&					coeff		- Coefficients initialized to first guess (weights)
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
				auto localPrediction = hyperplaneFormula(coeff);
				vector<double> dim;

				for (auto j{ 0U }; j < domains.size(); ++j) {
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
		auto logisticEstimation = [prediction](const vector<double>& dim) {
			return (1 / (1 + exp(-prediction(dim))));
		};
		return logisticEstimation;
	}

	/*
	*	Linear Discriminant Analysis 
	*
	*	Coefficient calculation:
	*	Coefficient i=1...(N-1) = 
	*   Coefficient0 = 
	*
	*	Estimated function calculation:
	*   f(values) -> 
	*
	*	Input:
	*	const vector<double>&			codomain	- Sulotions of the function (classes)
	*   const vector<vector<double>>&	domains		- The dimentios of the function
	*	vector<double>&					coeff		- Coefficients initialized to first guess (weights)
	*	dobule							alpha		- Change rate
	*	size_t							epochs		- Times to run through the dataset
	*
	*	Output:
	*	function<double(const vector<double>&)>		- Estimated function
	*/
	void LDAPreporcessing(const vector<double>& codomain, const vector<vector<double>>& domains, map<double, vector<double>>& mDomains);
	
	function<double(const vector<double>&)> linearDiscriminantAnalysis(const vector<double>& codomain, const vector<vector<double>>& domains, vector<double>& coeff, double alpha, size_t epochs) {
		map<double, vector<double>> mDomains;
		LDAPreporcessing(codomain, domains, mDomains);
		
		vector<double> mean;
		vector<double> probability;
		auto sumSqueredDiff{ 0.0 };

		for (auto it = mDomains.begin(); it != mDomains.end(); ++it) {
			// Find mean for each domain
			auto avg = cppmllib::average(it->second);
			mean.push_back(avg);
			
			// Find probablity for each class
			probability.push_back(static_cast<double>(it->second.size()) / codomain.size());
			
			// Find sum of squared diffs over all instances and classes
			for (auto i{ 0U }; i < it->second.size(); ++i) {
				double diff = it->second[i] - avg;
				sumSqueredDiff += diff * diff;
			}
		}

		auto variance = (1.0 / (domains[0].size() - mDomains.size())) * sumSqueredDiff;

		auto prediction = [mean, probability, variance](const vector<double> val) {
			auto pred{ -1.0 };
			auto currMaxDiscriminant{ DBL_MIN };

			for (auto i{ 0U }; i < mean.size(); ++i) {
				auto discriminant = val[0] * (mean[i] / variance) - ((mean[i] * mean[i]) / (2 * variance)) + log(probability[i]);
				if (discriminant > currMaxDiscriminant) {
					currMaxDiscriminant = discriminant;
					pred = i;
				}
			}
			return pred;
		};
		return prediction;
	}
	
	void LDAPreporcessing(const vector<double>& codomain, const vector<vector<double>>& domains, map<double, vector<double>>& mDomains) {
		// Sort domains by class
		for (auto i{ 0U }; i < codomain.size(); ++i) {
			mDomains[codomain[i]].push_back(domains[0][i]);	
		}
	}
}

