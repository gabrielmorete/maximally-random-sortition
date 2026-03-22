/**
 * @file optimization.h
 * @brief Implementation of the optimization algorithms.
 */

#include "main.h"

#pragma once



/**
 * @brief Calculates the squared L2 norm of a vector.
 * @param p The vector for which to calculate the norm.
 * @return The sum of the squares of the elements.
 */
long double getNormSq(const std::vector<long double> &p){
	long double d = 0;
	
	for (auto x : p)
		d += sq(x);

	return d;
}

/**
 * @brief Calculates the squared L2 distance between two vectors.
 * @param p The first vector.
 * @param q The second vector.
 * @return The sum of squared differences between elements of p and q.
 */
long double getL2Distance(std::vector<long double> &p, std::vector<long double> &q){
	long double d = 0;

	for (int i = 0; i < p.size(); i++)
		d += sq(p[i] - q[i]);

	return d;
}




/**
 * @brief Updates respondent weights using the Adam optimization algorithm.
 *
 * @details
 * This function performs a single step of Stochastic Gradient Ascent to maximize the 
 * Dual Objective function of the Maximum Entropy problem. 
 * 
 * The Objective:
 * We aim to find weights `u` such that the expected marginal probability of every 
 * respondent matches the target `targetP`. This is equivalent to maximizing:
 * 		g(lambda) = <targetP, \lambda> - log( \sum_{P is a panel} exp(lambda(P)) )
 * where `targetP - p` is the gradient of g (p is the vector of current marginals).
 *
 * The DP algorithm requires integer weights (`dType`). We make 
 * Wu_i = ceil(\exp(\lambda_i)) + 1$. 
 * We add 1 to ensure all weights are strictly positive (requirement for the DP).
 *
 * @param iter The current iteration number (1-based). Used for Adam bias correction.
 * @param u [Output] The vector of integer weights to be used in the next DP reweighting step.
 * @param targetP [Input] The target marginal probabilities (constraints).
 * @param p [Input] The current estimated marginal probabilities (from sampling).
 * @param lambda [Input/Output] The Lagrange multipliers (log-weights). Maintains state across iterations.
 * @param m [Input/Output] First moment vector (Momentum). Maintains state across iterations.
 * @param v [Input/Output] Second moment vector (Velocity). Maintains state across iterations.
 * @param alpha [Input] The current learning rate (typically decayed by the caller).
 * @return Always returns true.
 */
void updateWeightsAdam(
	int iter,
	vector<dType> &u, 
	const vector<floatType> &targetP,
	const vector<floatType> &p, 
	vector<floatType> &lambda,
	vector<floatType> &m, 
	vector<floatType> &v,
	floatType alpha){
	int n = p.size();

	// Standard Adam Hyperparameters (Pytorch: https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/include/torch/optim/adam.h)
	const floatType beta1 = 0.9;
	const floatType beta2 = 0.999;
	const floatType epsilon = 1e-8;
	const long double lg = log(1e7); // Scaling factor for dType conversion

	// Adam computation
	for (int i = 0; i < n; i++) {
		// Gradient Ascent on the Dual (Target - Current)
		floatType grad = targetP[i] - p[i];

		// Update biased first moment estimate
		m[i] = beta1 * m[i] + (1.0 - beta1) * grad;

		// Update biased second raw moment estimate
		v[i] = beta2 * v[i] + (1.0 - beta2) * sq(grad);

		// Compute bias-corrected estimates
		floatType m_hat = m[i] / (1.0 - pow(beta1, iter));
		floatType v_hat = v[i] / (1.0 - pow(beta2, iter));

		// Update Lambda
		lambda[i] += alpha * m_hat / (sqrt(v_hat) + epsilon);
	}

	// Converting weights into integral weights for the DP

	// Normalize Lambda to prevent overflow
	floatType lmx = *max_element(lambda.begin(), lambda.end());

	if constexpr (numeric_limits<dType>::is_integer) {
		// Convert log-weights (lambda) back to integer weights (u)
		for (int i = 0; i < n; i++)
			u[i] = 1 + (long long)ceill(expl((lambda[i] - lmx) + lg));
	} else {
		// Float/fractional behavior: continuous positive weights
		
		for (int i = 0; i < n; i++){
			floatType w = expl((lambda[i] - lmx) + lg);

			if (!isfinite((long double)w) || w <= 0) 
				w = numeric_limits<floatType>::min();

			u[i] = static_cast<dType>(w);
		}
	}
}



/**
 * @brief Computes a maximum entropy distribution generating target marginals
 * 
 * @details
 * This function orchestrates the optimization process to find a set of respondent weights 
 * such that the resultign weighted distrobuting generates a target marginal vector.
 * 
 * The problem is solved using a "Sample-Optimize-Reweight" loop. 
 * Since calculating the exact gradient is #P-Complete (requires summing over all panels), 
 * we estimate it via Monte Carlo sampling (`panel.getWeightedFrequencies`).
 * 
 * 
 * Heuristics and optimizations:
 * 
 * 	-Warmstart:
 * 		Before the loop starts, we confirm the maginals of the uniform distribution
 * 		Some respondents are "structurally rare" (few valid panels contain them) 
 * 		due to geometry/quotas, even if they have high target marginals.
 * 		We initialize weights using the ratio: 
 * 
 * 			lambda_0 = log(targetP_i) - log(uniformP_i).
 * 
 * - Adaptive Sampling:
 * 		We dynamically adjust the sample size  N_s based on the current error.
 * 		- High Error: We use fewer panels. The gradient direction is obvious, 
 * 			so high noise is acceptable. Speed is prioritized.
 * 		- Low Error We use many panels. As we approach the optimum, sampling 
 * 			noise dominates the gradient.
 * - Learning Rate Decay:
 * 		The learning rate `alpha` decays over time 1/(1 + kt} to 
 * 		to minimize variation.
 * 
 * Implementation
 * 
 * - Warmstart `lambda`
 * - Loop until convergence or max iterations:
 * 	1. Calculate `currentSamples` based on `currentError`.
 * 	2. Sample `currentSamples` panels from the DP.
 * 	3. Estimate current marginals `p`.
 * 	4. Step `updateWeightsAdam` to adjust `u`.
 * 	5. Call `panel.reweight(dp, u)` to recompute distribution with new weights.
 * @param `nPanels` The number of final panels to return after convergence.
 * @param `targetP` The target probability vector for each respondent.
 * @param `panel` The Panel object containing data and DP logic.
 * @return A vector of sampled panels (vectors of IDs) that generates the target marginals.
 */
vector< vector<int> > sampleFromTargetMarginals(int nPanels,
	vector<floatType> targetP,
	Panel &panel){


	int n = panel.getNumResp();
	int k = panel.getPanSize();
	

	/**
	 * Adaptive sampling
	 * We want the average person to appear ~100 times in the base sample.		 
	 */
	int basePanels = max(1'000, (int)(100.0 * (double)n / (double)k));
	int maxPanels = min(10'000, basePanels * 10); // Cap dynamic growth

	vector<dType> u(n, 1); // Weights of the respondents
	vector<floatType> lambda(n, 0); // log(u)
	vector<floatType> p(n); // Marginals given by current weights
	vector<floatType> m(n, 0.0); // Adam internal vector
	vector<floatType> v(n, 0.0); // Adam internal vector

	// Obtain the initial DP
	shared_ptr<CountSets> dp = panel.autoFeatureSelection();


	if (__verbose_mode){
		cerr << PURPLE <<" running Adam on instance"<< WHITE << endl;
		cerr << "\t Pool size: " << n << endl;
		cerr << "\t Panel size: " << k << endl;
		cerr << "\t Relative error tolerance: " << setprecision(3) << __adamRelativeEps << endl;
		cerr << "\t Absolute error tolerance: " << __adamAbsoluteEps << endl;
		cerr << "\t Max iterations: " << __adamMaxIter << endl;
		cerr << "\t Base alpha: " << __adamBaseAlpha << endl;
	}

	if (__test_mode) {
		cerr << BLUE << "base Panels: " << WHITE << basePanels << endl; 
		cerr << BLUE << "max Panels: " <<WHITE << maxPanels << endl;
	}

	/**
	 * Warmstart
	 * We want the average person to appear ~1000 times in the base sample.		 
	 */	
	int initialPanels = max(2'000, (int)(1000.0 * (double)n / (double)k));

	if (__test_mode)
		cerr << BLUE << "Sampling uniform marginals with : " << WHITE << initialPanels << BLUE << " panels." << WHITE << endl; 
	
	vector<long long> uniformFreq = panel.getUniformFrequencies(dp, initialPanels);
	
	for (int i = 0; i < n; i++){
		// Probability of seeing i in a uniform random panel
		p[i] = max<floatType>(1e-6, ((floatType)uniformFreq[i]) / initialPanels);
		
		if (targetP[i] < 1e-6)
			lambda[i] = -5.0; // effectively zero weight
		else
			lambda[i] = log(targetP[i]) - log(p[i]);
	}

	// Apply Initial Weights
	floatType lmx = *max_element(lambda.begin(), lambda.end());
	long double lg = log(1e12);
	for (int i = 0; i < n; i++) 
		u[i] = 1 + (long long)ceil(exp((lambda[i] - lmx) + lg));

	

	/**
	 * Optimization Loop 
	 */
	floatType targetPNrm = getNormSq(targetP);
	floatType l2Err = getL2Distance(p, targetP); // Start with uniform marginals
	floatType currentRelErr = l2Err / targetPNrm;;
	
	floatType bestL2Err = l2Err;
	floatType bestRelErr = currentRelErr;
	int bestIter = 0;
	vector<dType> bestU(n, 1); // Initialize with starting weights

	int iter = 1;
	while (l2Err > __adamAbsoluteEps and currentRelErr > __adamRelativeEps and iter <= __adamMaxIter){

		shared_ptr<CountSets> reWeightDP = panel.reweight(dp, u);

		/**
		 * Adaptive Sampling
		 * Increase panels as error decreases. When error is high, coarse panels are enough. 
		 * When error is low, we need precision.
		 */
		double errorFactor = (currentRelErr > 0.1) ? 0.0 : (0.1 / (currentRelErr + 0.001));
		int currentPanels = basePanels + (int)((double)basePanels * errorFactor);
		
		currentPanels = min(currentPanels, maxPanels);


		// Get Current Marginals
		vector<long long> respFreq = panel.getWeightedFrequencies(reWeightDP, currentPanels); 

		for (int i = 0; i < n; i++)
			p[i] = ((floatType) respFreq[i]) / (floatType)currentPanels;
		
		// Compute Error
		l2Err = getL2Distance(p, targetP);
		currentRelErr = l2Err / targetPNrm;


		if (l2Err < bestL2Err) {
			bestL2Err = l2Err;
			bestRelErr = currentRelErr;
			bestIter = iter;
			bestU = u; // Store the weights that generated this distribution
		}

		// Alpha Decay over time
		floatType currentAlpha = __adamBaseAlpha / (1.0 + 0.01 * iter);

		if (__test_mode){ 
			cerr << BLUE << "Iteration " << iter << WHITE << endl;
			cerr << "(panels: " << currentPanels << " alpha:" << fixed << setprecision(4) << currentAlpha << ") " << endl;
			cerr << "L2 Error: (Abs: " << l2Err << ") (Rel: " << currentRelErr << ")" << endl;
		}
			
		
		// Update Weights and distribution
		updateWeightsAdam(iter, u, targetP, p, lambda, m, v, currentAlpha);

		iter++;
	}


	if (__verbose_mode){
		cerr << "Best Solution Itr: " << bestIter << endl;
		cerr << "L2 Error: " << bestL2Err << " (Rel: " << bestRelErr << ")" << endl;
	}

	panel.reweight(dp, bestU);


	// Final Sampling
	auto pan = panel.samplingAlgorithm<true, true>(dp, nPanels, -1);
	return pan;
}