#pragma once

#include <functional>
#include <vector>
#include <map>

namespace cppmllib
{
	// linearAlgorithm function type
	typedef std::function<std::function<double(const std::vector<double>&)>(const std::vector<double>&, const std::vector<std::vector<double>>&, std::vector<double>&, double, size_t)> linearAlgorithm;
	
	std::function<double(const std::vector<double>&)> linearRegression(const std::vector<double>& codomain, const std::vector<std::vector<double>>& domains, std::vector<double>& coeff, double alpha = -1, size_t epochs = -1);
	std::function<double(const std::vector<double>&)> linearGradientDescent(const std::vector<double>& codomain, const std::vector<std::vector<double>>& domains, std::vector<double>& coeff, double alpha = 0.01, size_t epochs = 20);
	std::function<double(const std::vector<double>&)> logisticRegression(const std::vector<double>& codomain, const std::vector<std::vector<double>>& domains, std::vector<double>& coeff, double alpha = 0.3, size_t epochs = 6);
	std::function<double(const std::vector<double>&)> linearDiscriminantAnalysis(const std::vector<double>& codomain, const std::vector<std::vector<double>>& domains, std::vector<double>& coeff, double alpha = -1, size_t epochs = -1);
	
	// Functions not conforming to linearAlgorithm function type. Needed so std::tuple will distinguish between the two
	namespace overload {
		std::function<double(const std::vector<double>&)> linearDiscriminantAnalysis(const std::vector<double>& codomain, const std::vector<std::vector<double>>& domains, std::map<double, std::vector<double>>& mDomains);
	}
}