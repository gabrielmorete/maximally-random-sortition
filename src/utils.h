/**
 * @file utils.h
 * @brief A collection of general-purpose utility functions and macros.
 *
 * This file provides helpers for CSV reading, ostream overloads for various types,
 * type-agnostic conversion, and common debugging macros.
 */

#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <type_traits> 
#include <vector>
#include <boost/multiprecision/cpp_int.hpp>

#define RED "\033[31m"
#define BLUE "\033[34m"
#define WHITE "\033[0m"
#define YELLOW "\033[33m"
#define PURPLE "\033[45m"
#define GREEN "\033[32m"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>


/**
 * @brief Reads a CSV file into a 2D vector of strings.
 *
 * Each line in the file becomes a row in the vector, and each
 * comma-separated value becomes an element in the row vector.
 *
 * @param filePath The path to the CSV file.
 * @return A std::vector<std::vector<std::string>> containing the CSV data.
 * @throw std::runtime_error If the file cannot be opened.
 */
static inline void stripEOL(std::string& s) {
	while (!s.empty() && (s.back() == '\n' || s.back() == '\r'))
		s.pop_back();
}

static inline void stripBOM(std::string& s) {
	// UTF-8 BOM: 0xEF 0xBB 0xBF
	if (s.size() >= 3 &&
		static_cast<unsigned char>(s[0]) == 0xEF &&
		static_cast<unsigned char>(s[1]) == 0xBB &&
		static_cast<unsigned char>(s[2]) == 0xBF) {
		s.erase(0, 3);
	}
}

std::vector<std::vector<std::string>> readCSV(const std::string& filePath) {
	std::vector<std::vector<std::string>> data;
	std::ifstream file(filePath);

	if (!file.is_open()) {
		std::stringstream err_ss;
		err_ss << RED << "Error: " << WHITE << "could not open file " << filePath;
		throw std::runtime_error(err_ss.str());
	}

	std::string line;
	bool firstLine = true;

	while (std::getline(file, line)) {
		stripEOL(line);
		if (firstLine) {
			stripBOM(line);
			firstLine = false;
		}

		std::vector<std::string> row;
		std::stringstream ss(line);
		std::string cell;

		while (std::getline(ss, cell, ',')) {
			stripEOL(cell);
			row.push_back(cell);
		}

		// Preserve a trailing empty field if line ends with a comma.
		if (!line.empty() && line.back() == ',') {
			row.emplace_back("");
		}

		data.push_back(std::move(row));
	}

	return data;
}


/**
 * @brief Reads a vector of marginal probabilities from a CSV file.
 * 
 * @details Expects a CSV file where the first row is a header. 
 * Subsequent rows should contain the respondent ID in the first 
 * column and the marginal probability in the second.
 *
 * | ID | Marginal |
 * |----|----------|
 * | 0  | 0.5      |
 * | 1  | 0.2      |
 * 
 * * @param filePath The path to the CSV file containing the marginals.
 * @return A vector where index `i` corresponds to the marginal value 
 * 		of respondent `i`.
 * @throw runtime_error If a respondent ID is negative or exceeds 
 * the inferred vector size.
 */
std::vector<long double> readMarginals(const std::string& filePath){
	auto data = readCSV(filePath);

	int n = data.size() - 1; // Cut header

	std::vector<long double> marginals(n, 0);

	for (int i = 1; i < n; i++){
		int rId = stoi(data[i][0]);

		if (rId < 0 or rId >= n)
			throw std::runtime_error("Inconsistent vector of marginals");


		marginals[rId] = stod(data[i][1]);
	}

	return marginals;
}


/**
 * @brief Aggregates respondent marginals from a distribution over panels.
 * @details Expects a CSV file where the first row is a header. 
 * Each of the subsequent rows represents a valid panel.
 * The first column is the probability of the panel.
 * The k subsequent columns each represent a panel member.
 * 
 * | Probability | Member1_ID | Member2_ID | ... |
 * |-------------|------------|------------|-----|
 * | 0.1         | 5          | 10         | ... |
 * | 0.05        | 2          | 5          | ... |
 * 
 * * @param n The total number of respondents.
 * @param filePath The path to the CSV file containing the distribution.
 * @return A vector of size `n` where index `i` is the sum of probabilities 
 * 		of all panels containing respondent `i`.
 * @throw std::runtime_error If a respondent ID in the file is out of bounds [0, n).
 */
std::vector<long double> readMarginalsFromDist(const int n, const std::string& filePath){
	auto data = readCSV(filePath);

	int nPanels = data.size() - 1; // Cut header
	int panSz = data[0].size() - 1; // Size of the panel


	std::vector<long double> marginals(n, 0);

	for (int i = 1; i < nPanels; i++){
		long double q = stod(data[i][0]);

		for (int j = 1; j <= panSz; j++){
			int rId = stoi(data[i][j]);
			if (rId < 0 or rId >= n)
				throw std::runtime_error("Inconsistent distribution");
			
			marginals[rId] += q;
		}
	}

	return marginals;
}


/**
 * @brief Prints a list of panels in a specific CSV format.
 *
 * @details This function takes a vector of panels (where each panel is a
 * vector of respondent IDs) and prints them to the standard output,
 * one panel per line.
 *
 * @param panels A constant reference to a 2D vector of panels. Each inner
 * std::vector<int> represents a single panel.
 */
void writePanelsCSV(const std::vector< std::vector<int> > &panels){
	long double panProb = 1;
	panProb /= panels.size();

	int panSz = panels[0].size();

	std::cout << std::setprecision(7) << std::fixed;
	std::cout << "Panel Probability";
	for (int i = 0; i < panSz; i++)
		std::cout << "," << i + 1;
	std::cout << std::endl;		

	for (auto pan : panels){
		std::cout << panProb;

		sort(pan.begin(), pan.end());

		for (int rId : pan)
			std::cout << "," << rId;
		std::cout << std::endl;
	}	
}

/**
 * @brief Computes the square.
 * @param x The value to square.
 * @return The square of x.
 */
inline constexpr long double sq(long double x){ return x * x; }
inline constexpr double sq(double x){ return x * x; }





/**
 * @brief Provides a std::ostream overload for the __uint128_t type.
 *
 * This allows __uint128_t values to be printed directly to streams
 * like std::cout or std::cerr.
 *
 * @param os The output stream.
 * @param val The __uint128_t value to print (passed by value for modification).
 * @return A reference to the modified output stream.
 */
std::ostream& operator<<(std::ostream& os, __uint128_t val){
	if (val == 0){
		return os << "0";
	}

	std::string str;
	while (val > 0){
		str += '0' + (val % 10);
		val /= 10;
	}

	std::reverse(str.begin(), str.end());

	return os << str;
}

/**
 * @brief Provides a std::ostream overload for a vector of cpp_rational.
 *
 * This is a specialized overload that prints each rational
 * as a long double, separated by a space.
 *
 * @param os The output stream.
 * @param p The vector of cpp_rational values to print.
 * @return A reference to the modified output stream.
 */
std::ostream& operator<<(std::ostream& os, const std::vector<boost::multiprecision::cpp_rational>& p){
	// Use const auto& for efficiency (avoids copying large rational objects)
	for (const auto& x : p){
		os << x.convert_to<long double>() << ' ';
	}
	return os;
}

/**
 * @brief Provides a generic std::ostream overload for std::vector.
 *
 * @tparam T The type of element in the vector.
 * @param os The output stream.
 * @param p The vector of values to print.
 * @return A reference to the modified output stream.
 */
template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& p){
	for (const auto& x : p){
		os << x << ' ';
	}
	return os;
}

/**
 * @brief Converts a value to a specified type U, handling both extended-precision
 * types (with .convert_to) and native types.
 *
 * This template function uses `if constexpr` (C++20) to check if the
 * .convert_to<U>() member function exists.
 * - If it exists (e.g., for Boost.Multiprecision types), it calls it.
 * - If not (e.g., for native types like long long), it uses static_cast<U>.
 *
 * @tparam U The target type to convert to.
 * @tparam T The source type to convert from.
 * @param val The value to convert.
 * @return The value converted to type U.
 */
template<typename U, typename T>
U generic_convert(const T& val){
	if constexpr (requires{ val.template convert_to<U>(); }){
		return val.template convert_to<U>();
	} else{
		return static_cast<U>(val);
	}
}




/*
 * Debug
 */
#define dbg(x) cerr << #x << " = " << x << endl
#define chapa cerr << "Oi meu chapa" << endl
