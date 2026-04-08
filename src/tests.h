/**
 * @file tests.h
 * @brief Defines functions for running correctness tests and benchmarks.
 *
 * This file includes functions to read test data, verify the dynamic programming
 * results against known outputs, compare sampling accuracy against a brute-force
 * method, and run performance benchmarks.
 */

#pragma once


#include "main.h"
#include "panel.h"

/**
 * @brief Reads test case data from a specified file.
 *
 * Each line in the file represents a test case. The line is split into
 * space-separated parts. Expected format: feature1 feature2 ... expected_count
 *
 * @param filePath The path to the test file.
 * @return A std::vector<std::vector<std::string>> where each inner vector
 * represents a single test case row from the file.
 * @throw std::runtime_error If the file cannot be opened.
 */
std::vector<std::vector<std::string>> readTest(const std::string& filePath){
	std::ifstream file(filePath);
	std::vector<std::vector<std::string>> data;
	std::string line;

	if (!file.is_open()){
		std::stringstream err_ss;
		err_ss << RED << "Error: " << WHITE << "could not open test file " << filePath;
		throw std::runtime_error(err_ss.str());
	}

	while (std::getline(file, line)){
		std::istringstream iss(line);
		std::vector<std::string> entry;
		std::string part;
		while (iss >> part){
			entry.push_back(part);
		}
		if (!entry.empty()){
			data.push_back(entry);
		}
	}

	file.close();
	return data;
}

/**
 * @brief Runs a suite of correctness tests for the DP counting algorithm.
 *
 * Loads datasets and corresponding expected results from test files.
 * For each test case, it checks:
 * 1. The panel count matches the expected count.
 * 2. For 2-feature cases, the count is symmetric (order doesn't matter).
 * 3. A simple reweighting scenario yields the expected scaled count.
 * Prints PASSED or FAILED for each test and a summary at the end.
 */
void runTests(){

	if (std::numeric_limits<dType>::is_integer == 0)
		throw std::runtime_error("test funtion requires integer dType.");
	


	const std::vector<std::string> datasetNames ={
		// "simple_ABC_2", "infeasible_ABC_2", "full_ABC_4", 
		"sf_b_20", "mass_a_24", "hd_30", "obf_30", "sf_a_35", "sf_d_40",
		"sf_c_44", "cca_75", "sf_e_110"
	};

	const std::vector<int> panelSizes ={
		// 2, 2, 4, 
		20, 24, 30, 30, 35, 40, 44, 75, 110
	};

	std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
	std::uniform_int_distribution<int> wgtDist(2, 1000); // For reweighting test

	std::cout << BLUE << "\n--- Testing DP Correctness ---" << WHITE << std::endl;
	int totalDpTests = 0;
	int passedDpTests = 0;

	for (size_t i = 0; i < datasetNames.size(); ++i){
		const std::string& name = datasetNames[i];
		const int panelSize = panelSizes[i];
		std::string filePath = "../data_max_entropy/" + name + "/";

		std::cout << "\n----- " << name << " -----" << std::endl;

		try{
			Panel panel(panelSize, filePath + "categories.csv", filePath + "respondents.csv");
			auto tests = readTest("test_files/" + name + ".out");

			// --- Loop Through Test Cases for this Dataset ---
			for (const auto& testCase : tests){
				totalDpTests++;

				if (testCase.empty()) continue; // Skip empty lines/cases

				std::vector<int> features;
				std::string featureStr;
				// Parse features (all elements except the last)
				for (size_t j = 0; j < testCase.size() - 1; ++j){
					if (!testCase[j].empty()){ // Basic validation
						features.push_back(testCase[j][0] - 'a'); // Assumes 'a', 'b', ...
						featureStr += testCase[j] + " ";
					}
				}

				cpp_int expectedCount(testCase.back());


				std::cout << "  Test: " << std::left << std::setw(15) << featureStr;

				bool mainPass = false;
				bool symmetryPass = true;
				bool reweightPass = true;
				std::stringstream errorDetails;

				// Count Check
				dType foundCount = panel.count(features);
				if (static_cast<cpp_int>(foundCount) == expectedCount){
					mainPass = true;
				} else{
					errorDetails << "\n    [Count Mismatch] Expected: " << expectedCount
								 << " | Found: " << foundCount;
				}

				// Symmetry Check
				if (features.size() == 2){
					std::vector<int> reversedFeatures = features; // Make a copy
					std::reverse(reversedFeatures.begin(), reversedFeatures.end());
					dType symmetricCount = panel.count(reversedFeatures);
					if (symmetricCount != foundCount){
						symmetryPass = false;
						errorDetails << "\n    [Symmetry Error] Reversed count: " << symmetricCount;
					}
				}

				// Reweighting Check
				// Assign a uniform random weight 'w' to all respondents.
				// The total count should scale by w^(panelSize).
				if (mainPass){ // Only run reweight test if main count was correct
					dType w = wgtDist(rng); // Generate a random weight
					std::vector<dType> mu(panel.getNumResp(), w);
					dType expectedReweighted = foundCount; // Start with original count
					for (int f = 0; f < panel.getPanSize(); ++f){

						if (std::numeric_limits<dType>::is_bounded &&
							expectedReweighted > std::numeric_limits<dType>::max() / w){
							 errorDetails << "\n    [Reweigh Warning] Potential overflow calculating expected count.";
							 expectedReweighted = 0; // Mark as potentially invalid
							 break;
						}
						expectedReweighted *= w;
					}


					dType actualReweighted = panel.reweight(features, mu);

					if (expectedReweighted != 0 && expectedReweighted != actualReweighted){
						 reweightPass = false;
						 errorDetails << "\n    [Reweigh Error w=" << w << "] Expected: " << expectedReweighted
									  << " | Found: " << actualReweighted;
					}
					panel.resetWeights(); // Reset for the next test case
				} else{
					reweightPass = false; // Cannot test reweight if base count failed
					errorDetails << "\n    [Reweigh Skipped] Base count failed.";
				}


				if (mainPass && symmetryPass && reweightPass){
					std::cout << GREEN << "PASSED" << WHITE << std::endl;
					passedDpTests++;
				} else{
					std::cout << RED << "FAILED" << WHITE << errorDetails.str() << std::endl;
				}
			}
		} catch (const std::exception& e){
			 std::cerr << RED << "ERROR processing dataset " << name << ": " << e.what() << WHITE << std::endl;
		}

	}

	std::cout << BLUE << "\n--- DP Test Summary: "
			  << passedDpTests << " / " << totalDpTests << " passed ---" << WHITE << std::endl;


	/*
		Test sampling 
	*/         


   
	std::cout << BLUE << "\n--- Testing Sampling Accuracy ---" << WHITE << std::endl;

	const std::vector<std::string> samplingDatasetNames = {//"simple_ABC_2", "ABC_2", "full_ABC_4", 
															"mass_a_24"};
	const std::vector<int> samplingPanelSizes = {//2, 2, 4, 
												24};
	const long long nPanelsSampling = 200'000; // Number of panels for comparison
	const double samplingTolerance = 0.05;     // 5% relative tolerance

	for (size_t i = 0; i < samplingDatasetNames.size(); ++i){
		const std::string& name = samplingDatasetNames[i];
		const int panelSize = samplingPanelSizes[i];
		std::string filePath = "../data_max_entropy/" + name + "/";

		std::cerr << "\n ----- " << name << " ----- " << std::endl; // Use cerr for progress

		try{
			Panel panel(panelSize, filePath + "categories.csv", filePath + "respondents.csv");

			std::vector<long long> bruteFrq = panel.bruteRespFrequency(nPanelsSampling);
			std::shared_ptr<CountSets> dp = panel.autoFeatureSelection(); // Get the DP structure first
			 if (!dp || dp->getAllSets() == 0){
				 std::cerr << YELLOW << "Skipping sampling test for " << name << ": DP is infeasible or failed." << WHITE << std::endl;
				 continue;
			 }
			std::vector<long long> algoFrq = panel.getUniformFrequencies(dp, nPanelsSampling);

			int nRespondents = panel.getNumResp();
			double maxRelErr = 0.0;
			double sumRelErr = 0.0;
			int nonZeroRespCount = 0; // Respondents present in brute force

			// Verify total counts match (should be nPanels * panelSize)
			long long expectedTotalSamples = (long long)nPanelsSampling * panelSize;
			long long totalSampleBrute = std::accumulate(bruteFrq.begin(), bruteFrq.end(), 0LL);
			long long totalSampleAlgo = std::accumulate(algoFrq.begin(), algoFrq.end(), 0LL);

			bool totalsMatch = (totalSampleBrute == expectedTotalSamples && totalSampleAlgo == expectedTotalSamples);


			// Calculate relative error only for respondents seen in brute force
			for (int j = 0; j < nRespondents; ++j){
				if (bruteFrq[j] > 0){
					nonZeroRespCount++;
					long long diff = bruteFrq[j] - algoFrq[j];
					// Use static_cast<double> for floating-point division
					double relErr = static_cast<double>(std::abs(diff)) / bruteFrq[j];

					sumRelErr += relErr;
					if (relErr > maxRelErr){
						maxRelErr = relErr;
					}
				} else if (algoFrq[j] != 0){
					// Algorithm found a respondent brute force didn't. This indicates
					// either insufficient brute force samples or an algorithm error.
					// Mark as infinite error for failure condition.
					maxRelErr = std::numeric_limits<double>::infinity();
					totalsMatch = false; // Force failure if this happens
					std::cerr << YELLOW << "Warning: Algorithm found respondent " << j
							  << " (" << algoFrq[j] << " times) but brute force did not." << WHITE << std::endl;
				}
			}

			double meanRelErr = (nonZeroRespCount > 0) ? (sumRelErr / nonZeroRespCount) : 0.0;

			// --- Report Result ---
			std::cout << std::fixed << std::setprecision(2);
			std::cout << "Max Relative Error:  " << std::setw(8) << maxRelErr * 100.0 << " %\n";
			std::cout << "Mean Relative Error: " << std::setw(8) << meanRelErr * 100.0 << " %\n";
			std::cout << "Total Samples Expected: " << expectedTotalSamples << "\n";
			std::cout << "Total Samples (Brute):  " << totalSampleBrute << "\n";
			std::cout << "Total Samples (Algo):   " << totalSampleAlgo << "\n";


			if (totalsMatch && maxRelErr <= samplingTolerance){
				std::cout << GREEN << "PASSED" << WHITE << " (within " << samplingTolerance * 100.0 << "% tolerance)" << std::endl;
			} else{
				std::cout << RED << "FAILED" << WHITE << " (Tolerance: " << samplingTolerance * 100.0 << "%";
				if (!totalsMatch) std::cout << ", Totals mismatch!";
				if (maxRelErr > samplingTolerance) std::cout << ", Max error exceeded!";
				 std::cout << ")" << std::endl;
			}
		 } catch (const std::exception& e){
			 std::cerr << RED << "ERROR processing sampling for " << name << ": " << e.what() << WHITE << std::endl;
		 }

	} // End loop over sampling datasets
}

/**
 * @brief Runs performance benchmarks on selected datasets.
 *
 * For each specified dataset, it measures:
 * 1. Time taken for DP setup (via `autoFeatureSelection`).
 * 2. Time taken to sample a fixed number of panels (`nPanels`).
 * Calculates and prints average time per panel, average tries per panel,
 * the DP subspace count, and an estimate of the total number of valid panels.
*/
void runBenchmark(){
	// const std::vector<std::string> datasetNames ={"modif_sf_e_110"};
	// const std::vector<int> panelSizes ={110};


	const std::vector<std::string> datasetNames ={
		"sf_b_20", "mass_a_24",  "sf_a_35", "sf_d_40", "sf_c_44",
	};

	const std::vector<int> panelSizes ={
		20, 24, 35, 40, 44
	};



	const long nPanelsBenchmark = 10'000; // Number of panels to sample for timing

	std::cout << "DP timeout setting: " << __dp_timeout << " minutes." << std::endl;

	for (size_t i = 0; i < datasetNames.size(); ++i){
		const std::string& name = datasetNames[i];
		const int panelSize = panelSizes[i];
		std::cout << "\n----- " << name << " -----" << std::endl;
		std::string filePath = "../data_panelot/" + name + "/";

		try{
			Panel panel(panelSize, filePath + "categories.csv", filePath + "respondents.csv");

			// Measure DP Setup Time
			auto dp_start = std::chrono::high_resolution_clock::now();
			std::shared_ptr<CountSets> dp = panel.autoFeatureSelection(); // This runs the DP
			auto dp_end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> dp_duration = dp_end - dp_start;

			// Check if DP was successful and feasible
			if (!dp){
				std::cout << "  Result: DP auto-selection failed or timed out." << std::endl;
				continue;
			}
			dType dp_count = dp->getAllSets();
			if (dp_count == 0){
				std::cout << "  Result: Problem is infeasible (0 panels found in DP)." << std::endl;
				continue;
			}

			// Measure Sampling Time
			auto sample_start = std::chrono::high_resolution_clock::now();
			std::vector<long long> respFreq = panel.getUniformFrequencies(dp, nPanelsBenchmark);
			auto sample_end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> sample_duration = sample_end - sample_start;

			long long totalSamples = panel.__number_of_tries; // Get samples from Panel member

			double avgTimePerPanel = (nPanelsBenchmark > 0) ? (sample_duration.count() / nPanelsBenchmark) : 0.0;
			double triesPerPanel = (nPanelsBenchmark > 0) ? (static_cast<double>(totalSamples) / nPanelsBenchmark) : 0.0;

			// Estimate total panels using rejection sampling principle:
			// (Valid Found / Total Tries) approx = (Actual Total Valid / DP Subspace Count)
			// => Actual Total Valid approx = (Valid Found / Total Tries) * DP Subspace Count
			// => Actual Total Valid approx = (nPanelsBenchmark / totalSamples) * dp_count
			dType estimatedTotalPanels = 0;
			if (totalSamples > 0){
				 // Use floating point for intermediate calculation to avoid premature overflow/truncation
				 // Convert back to dType at the end if necessary, or keep as double/long double
				 long double estimate_fp = (static_cast<long double>(nPanelsBenchmark) / totalSamples) * generic_convert<long double>(dp_count);
				 // Safely convert back if possible, or handle potential overflow
				 // For now, just outputting the long double estimate
				 estimatedTotalPanels = static_cast<dType>(estimate_fp); // Note: May overflow/truncate!
			}


			// Format active features string
			const auto& activeFeatures = dp->getActiveFeat();
			std::stringstream featureStr;
			featureStr << activeFeatures.size() << " features: ";
			for (size_t j = 0; j < activeFeatures.size(); ++j){
				featureStr << (char)('a' + activeFeatures[j])
						   << (j == activeFeatures.size() - 1 ? "" : " ");
			}

			std::cout << std::fixed;
			std::cout << "  DP Setup Time:         " << std::setw(10) << std::setprecision(4) << dp_duration.count() << " s  ("
					  << featureStr.str() << ")\n";
			std::cout << "  Total Sampling Time:   " << std::setw(10) << std::setprecision(4) << sample_duration.count() << " s  (for "
					  << nPanelsBenchmark << " panels)\n";
			std::cout << "  -------------------------------------------\n";
			std::cout << "  Avg. Time / Panel:     " << std::setw(10) << std::setprecision(4) << avgTimePerPanel * 1000 << " ms\n";
			std::cout << "  Avg. Tries / Panel:    " << std::setw(10) << std::setprecision(2) << triesPerPanel << "\n";
			std::cout << "  DP Subspace Count:     " << dp_count << "\n";
			std::cout << "  Total Samples (Tries): " << totalSamples << "\n";
			std::cout << "  Estimated Total Panels: " << estimatedTotalPanels << " (Note: Estimate depends on sampling rate)" << std::endl;

			

		} catch (const std::exception& e){
			 std::cerr << RED << "ERROR processing benchmark for " << name << ": " << e.what() << WHITE << std::endl;
		}

	}

	return;
}
