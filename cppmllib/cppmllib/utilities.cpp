#include <vector>
#include <numeric>
#include <functional>

#include "utilities.h"

namespace cppmllib
{
	double average(const std::vector<double>::const_iterator& cbegin, const std::vector<double>::const_iterator& cend)
	{
		auto n_elements = std::distance(cbegin, cend);

		THROW_IF_ZERO(n_elements);

		double sum = std::accumulate(cbegin, cend, 0);
		return (sum / static_cast<double>(n_elements));
	}

    // RMSE calculate an error score for the prediction mades by estfunc
	double rootMeanSquareError(const std::vector<double>& output, const std::vector<double>& input,
		const std::function<double(const double&)>& estfunc)
	{
		double sum{ 0 };

		THROW_IF_ZERO(output.size());
		THROW_IF_NOT_EQUAL(output.size(), input.size());		

		for (size_t i{ 0 }; i < output.size(); ++i)
		{
			double error = estfunc(input.at(i)) - output.at(i);
			sum += error * error;
		}

		return sqrt(sum / output.size());
	}
}