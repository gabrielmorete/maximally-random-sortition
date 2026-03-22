/**
 * @file main.cpp
 * @brief Implementation of Maximum Entropy sortition.
 *
 * This file implements the command-line interface for the algorithm.
 * It orchestrates data loading, preprocessing, and the execution of the algorithm
 * 	- Exact and approximate counting of valid panels (DP).
 * 	- Uniform and weighted rejection sampling.
 * 	- Maximum Entropy optimization to match target marginal distributions.
 *
 * Usage:
 * ./main -path <data_dir> -size <n> [options]
 */

#include "src/main.h"
#include "src/tests.h"
#include "src/panel.h"
#include "src/utils.h"
#include "src/optimization.h"


/**
 * @class InputParser
 * @brief A class for parsing command-line arguments.
 */
class InputParser {
	private:
		vector<string> tokens;

	public:
		/**
		 * @brief Constructs the parser from main's arguments.
		 * @param argc Argument count.
		 * @param argv Argument values.
		 */
		InputParser(int &argc, char **argv){
			for (int i = 1; i < argc; i++)
				this->tokens.push_back(string(argv[i]));
		}

		/** 
		 * @brief Retrieves the value associated with a specific option.
		 * Example: If args are `-size 100`, `getCmdOption("-size")` returns "100".
		 * @param option The flag string (e.g., "-s").
		 * @return The value following the flag, or an empty string if not found.
		 */
		const string& getCmdOption(const string &option) const {
			
			auto itr = find(this->tokens.begin(), this->tokens.end(), option);
			if (itr != this->tokens.end() && ++itr != this->tokens.end())
				return *itr;
			
			static const string emptyString("");
			return emptyString;
		}

		/** @brief Checks if a specific flag is present.
		 * Example: `cmdOptionExists("-verbose")` returns true if present.
		 * @param option The flag string to search for.
		 * @return `true` if the flag exists, `false` otherwise.
		 */
		bool cmdOptionExists(const string &option) const {
			return find(this->tokens.begin(), this->tokens.end(), option) != this->tokens.end();
		}

		/**
		 * @brief Collects multiple consecutive values following a flag.
		 * For arguments like `-count a b c`. It stops reading when it
		 * encounters the next string starting with '-'.
		 * @param option The flag string starting the sequence.
		 * @return A vector of strings containing the values found after the flag.
		 */
		vector<string> getMultipleValues(const string &option) const {
			vector<string> values;
			auto itr = find(this->tokens.begin(), this->tokens.end(), option);
			if (itr == this->tokens.end()) 
				return values;

			itr++; // Move past the flag
			while (itr != this->tokens.end() && itr->at(0) != '-'){
				values.push_back(*itr);
				itr++;
			}
			return values;
		}
};

/**
 * @brief Validates that a file exists at the given path.
 * @param path The path of the file.
 * @param desc A description of the file (for error logging).
 * @throw std::runtime_error If the file does not exist.
 */
void checkFileExists(const string& path, const string& desc){
	if (!std::filesystem::exists(path))
		throw runtime_error("File not found (" + desc + "): " + path);
}

/**
 * @brief Prints the usage instructions and available options to stderr.
 */
void printUsage(){
	cerr << "Usage: ./main -path <dir> -size <n> [options]\n"
			  << "Options:\n"
			  << "  -path <path>    : Path to data directory (must contain categories.csv and respondents.csv)\n"
			  << "  -size <n>       : Target panel size (required)\n"
			  << "  -sample <n>     : Number of panels to sample\n"
			  << "  -count <f1 f2>  : Count valid panels for specific features (e.g. -count a b)\n"
			  << "  -threads <n>    : Number of threads (default: 1)\n"
			  << "  -preprocess <t> : Enable preprocessing with threshold t\n"
			  << "  -tDist <file>   : Target distribution file\n"
			  << "  -test           : Enable test mode\n"
			  << "  -verbose        : Enable verbose output\n"
			  << "  -runTests       : Run internal code tests\n"
			  << "  -runBenchmark   : Run benchmarks\n";
}

signed main(int argc, char *argv[]){
	InputParser input(argc, argv);

	/** Help information*/
	if (input.cmdOptionExists("-h") || input.cmdOptionExists("--help")){
		printUsage();
		return 0;
	}

	if (input.cmdOptionExists("-verbose"))
		__verbose_mode = true;

	if (input.cmdOptionExists("-test"))
		__test_mode = true;
	
	string threadStr = input.getCmdOption("-threads");
	if (!threadStr.empty()){
		try {
			__num_threads = stoi(threadStr);
			#ifdef _OPENMP
				omp_set_num_threads(__num_threads);
			#endif
		} catch (...){
			cerr << RED << "Invalid thread count." << WHITE << endl;
			return 1;
		}
	}

	// Now, seed one rng for each thread
	__thread_rngs.resize(__num_threads);

	uint64_t seed0 = (uint64_t)chrono::steady_clock::now().time_since_epoch().count();
	std::mt19937_64 seed_rng(seed0);

	__thread_rngs[0].seed(seed0);
	for (int t = 1; t < __num_threads; t++)
		__thread_rngs[t].seed(seed_rng());
	

	if (input.cmdOptionExists("-preprocess")){
		__preprocess = true;
		string thr = input.getCmdOption("-preprocess");
		if (!thr.empty()) 
			__preprocess_theshold = stod(thr);
	}

	if (input.cmdOptionExists("-runBenchmark")){
		runBenchmark();
		return 0;
	}

	if (input.cmdOptionExists("-runTests")){
		runTests();
		return 0;
	}

	try {
		string filePath = input.getCmdOption("-path");
		string pSzStr = input.getCmdOption("-size");

		if (filePath.empty()){
			throw std::runtime_error("Missing required argument: -path");
		}
		if (pSzStr.empty()){
			throw std::runtime_error("Missing required argument: -size");
		}

		int pSz = stoi(pSzStr);
		if (pSz <= 0) 
			throw std::runtime_error("Panel size must be positive.");

		// Ensure path ends with / for concatenation
		if (filePath.back() != '/') 
			filePath += "/"; 
		
		string catFile = filePath + "categories.csv";
		string respFile = filePath + "respondents.csv";
		
		checkFileExists(catFile, "Categories Data");
		checkFileExists(respFile, "Respondents Data");

		// Initialize Panel
		Panel panel(pSz, catFile, respFile);


		string tDistPath = input.getCmdOption("-tDist");
		bool targetMarginals = !tDistPath.empty();
		
		bool doSample = input.cmdOptionExists("-sample");
		
		// Parse Count Features
		vector<string> countArgs = input.getMultipleValues("-count");
		vector<int> features;
		for(const auto& s : countArgs){
			 if (s.length() == 1 && s[0] >= 'a' && s[0] <= 'z'){
				 features.push_back(s[0] - 'a');
			 }
		}
		bool doCount = !features.empty();

		// Handle Single Feature Edge Case
		if (features.size() == 1){
			cerr << PURPLE << "Warning:" << WHITE << " Single feature count not supported. Duplicating feature." << endl;
			features.push_back(features.back());
		}

		string nStr = input.getCmdOption("-sample");
		int nPanels = 1;
			if (!nStr.empty()) 
				nPanels = stoi(nStr);

		if (nPanels < 1){
			cerr << RED << "ERROR:" << WHITE << " number of panels must be positive." << endl;
			return 1;
		}	



		if (targetMarginals){
			checkFileExists(tDistPath, "Target Distribution");

			auto marginals = readMarginalsFromDist(panel.getNumResp(), tDistPath);
			auto pan = sampleFromTargetMarginals(nPanels, marginals, panel);
			writePanelsCSV(pan);
			return 0;
		}

		if (doCount && doSample){
			dType numPanel = panel.count(features);
			cout << "# of panels : " << numPanel << endl;
			auto pan = panel.sampleUniformPanel(features);
			// panel.printPanel(pan);
		}
		else if (doCount){
			dType numPanel = panel.count(features);
			cout << "# of panels : " << numPanel << endl;
		}
		else if (doSample){

			auto dp_start = std::chrono::high_resolution_clock::now();
			auto dp = panel.autoFeatureSelection();
			auto dp_end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> dp_duration = dp_end - dp_start;

			

			auto sampling_start = std::chrono::high_resolution_clock::now();
			auto pan = panel.samplingAlgorithm<false, true>(dp, nPanels, -1);
			auto sampling_end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> sampling_duration = sampling_end - sampling_start;

			if (__test_mode) 

			if (__test_mode){ 
				cerr << BLUE << "DP Setup Time: " << WHITE << dp_duration.count() << "(s)" << endl;
				cerr << BLUE << "Sampling Time: " << WHITE << sampling_duration.count() << "(s)" << std::endl;
			}	
			// writePanelsCSV(pan);
		}

	} catch (const std::exception& e){
		cerr << RED << "Error: " << WHITE << e.what() << endl;
		return 1;
	}

	return 0;
}
