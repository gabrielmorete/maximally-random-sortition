/**
 * @file main.h
 * @brief Core definitions, type aliases, and configuration.
 *
 * Includes necessary headers, defines fundamental types like dType (for counts) and
 * fracType (for fractional calculations), and sets up global flags and constants
 * controlling program behavior like verbosity, testing modes, threading, and timeouts.
 */

#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <cstdint> 
#include <cassert>
#include <algorithm>
#include <chrono>
#include <tuple>
#include <memory>
#include <cmath>
#include <thread>
#include <atomic>  
#include <type_traits> 
#include <limits> 
#include <omp.h>
#include <string>
#include <filesystem> 
#include <iomanip>
#include <random>

#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp> 
#include <boost/functional/hash.hpp>
#include <boost/random.hpp> 
#include <boost/container/flat_map.hpp>
#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>
#include <boost/container/small_vector.hpp>

#include "utils.h" 

using namespace std;
using namespace boost::multiprecision;
using namespace boost::random;

typedef cpp_int dType;
typedef cpp_rational fracType;
typedef long double floatType; 

/** @brief Vector that is optimized for small sizes, 5 is a arbitrary choice. */
using FeatVec = boost::container::small_vector<uint8_t, 5>;
// using FeatVec = vector<int>;


/**
 * Instance limits
 * */

/** @brief Maximum number of features considered in a DP state (including feature 0). 
	Note that, there can be more features in the instance, the DP just won't accept it
*/
static constexpr uint8_t __maxFeatures = 8;


/* Maximum number of distinct values per feature. */
static constexpr uint8_t __maxFeatureValues = 64;

/* Maximum number of panel members. */
// 65'535 due to the use of uint16_t in mask.h


/**
 * Global Output Flags
 * */

/** @brief If true, enables printing extra informational messages during execution. */
bool __verbose_mode = 0;

/** @brief If true, enables printing detailed diagnostic information. */
bool __test_mode = 0;

/** @brief Structures output to be used with the vizualization tool at panelot */
bool __panelot_output = 0;

/**
 * DP constants
 * */

/** @brief If true, enables data preprocessing heuristics before running the DP. */
bool __preprocess = 0;
double __preprocess_theshold = 0;


/** @brief Timeout duration in minutes for the dynamic programming calculation. */
int __dp_timeout = 2;

/**
 * @brief Atomic flag set to true by a watchdog thread when the DP timeout is reached.
 * Checked within the DP algorithm to trigger early termination.
 */
atomic<bool> __stopDPTimeout(false);
/**
 * @brief Atomic flag set to true when the DP calculation finishes successfully.
 * Signals the watchdog thread to terminate early.
 */
atomic<bool> __DPFinished(false);


/**
 * Sampling constants
 * */


/** Global counter with the number of panels found. */
long long __number_of_panels_found = 0;

/** @brief Flush local sampling statistics every this many tries. */
static constexpr long long __sampling_update_tries = 200'000;

/** @brief Minimum time between non-final sampling-info prints. */
static constexpr int __sampling_update_seconds = 2;

/** @brief Number of parallel threads to use (e.g., for OpenMP sections). */
int __num_threads = 1;

/** @brief Per-thread RNG, seeded once for reproducibility*/
vector<std::mt19937_64> __thread_rngs;


/**
 * Optimization constants
 * */

/** @brief Relative error tolerance for the Adam optimizer. */
const floatType __adamRelativeEps = 0.001;

/** @brief Absolute error tolerance for the Adam optimizer. */
const floatType __adamAbsoluteEps = 0.001;

/** @brief Maximum number of iterations for the Adam optimizer.  */
const int __adamMaxIter = 200;

/** @brief Maximum number of minutes for the Adam optimizer.  */
const int __adamTimeout = 20;

/**  @brief Initial learning rate for the Adam optimizer. */
const floatType __adamBaseAlpha = 0.05;

