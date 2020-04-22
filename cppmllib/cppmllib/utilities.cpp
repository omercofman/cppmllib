
#include "utilities.h"

#include <vector>
#include <numeric>
#include <functional>

using std::vector;
using std::function;

namespace cppmllib
{
	double average(const vector<double>::const_iterator& cbegin, const vector<double>::const_iterator& cend) {
		auto n_elements = std::distance(cbegin, cend);
		THROW_IF_ZERO(n_elements);

		auto sum = std::accumulate(cbegin, cend, 0.0);
		return (sum / n_elements);
	}

	double average(const vector<double>& vec) {
		return average(vec.cbegin(), vec.cend());
	}

    /*
	*	RMSE calculate an error score for the prediction mades by estfunc
	*	Calcualted as: SQRT((SUM i=0...(N-1)[xi - yi])^2 / N)
	*/
	double rootMeanSquareError(const vector<double>& codomain, const vector<double>& dim, const function<double(const vector<double>&)>& estfunc) {
		auto sum{ 0.0 };

		THROW_IF_ZERO(codomain.size());
		THROW_IF_NOT_EQUAL(codomain.size(), dim.size());		

		for (auto i{ 0U }; i < codomain.size(); ++i) {
			auto error{ estfunc({ dim[i] }) - codomain[i] };
			sum += error * error;
		}

		return sqrt(sum / codomain.size());
	}
}