/**
 * @file panel.h
 * @brief Implements DP counting class (CountSets) and the Panel class.
 *
 * This file contains the primary logic for panel counting, reweighting, 
 * sampling, and auto-feature selection.
 */

#pragma once

#include "main.h"
#include "mask.h"
#include "state.h"
#include "data.h"


/**
 * @brief Runs a watchdog thread to enforce a time limit on the DP computation.
 *
 * This function runs in a separate thread. It sleeps and checks global atomic
 * flags. If the specified duration is exceeded before the DP finishes
 * (signaled by `__DPFinished`), it sets `__stopDPTimeout` to true, which
 * instructs the DP algorithm to throw a `DpTimeoutException`.
 *
 * @param maxDuration The maximum allowed runtime in minutes.
 * @note Relies on global atomic flags: `__DPFinished` (read) and `__stopDPTimeout` (write).
 */
void runWatchdog(int maxDuration){
	auto timeout = chrono::minutes(maxDuration);
	auto startTime = chrono::steady_clock::now();

	// Wake up every 1 seconds to check the flags
	while (chrono::steady_clock::now() - startTime < timeout) {

		if (__DPFinished) // DP is done before the timeout, exit early
			return;
	   
		this_thread::sleep_for(chrono::seconds(1));
	}

	__stopDPTimeout = true; // Timeout, stop the DP
	if (__test_mode)
		cerr << BLUE << "Watchdog:" << WHITE << " timeout triggered." << WHITE << endl;
}


/**
 * @class DpTimeoutException
 * @brief Custom exception thrown when the DP calculation is interrupted by the watchdog.
 */
class DpTimeoutException : public exception{
public:
	const char* what() const noexcept override {
		return "DP calculation timed out.";
	}
};

/**
 * @class DpTimeoutException
 * @brief Custom exception thrown when the DP calculation is interrupted by the watchdog.
 */
class DpStateException : public exception{
public:
	const char* what() const noexcept override {
		return "DP exceeded the maximum number of states.";
	}
};

/**
 * @class CountSets
 * @brief An implementation of a DP algorithm to count the number of feasible panels.
 *
 * This class solves the generalized counting problem for a given PanelData object,
 * respondent weights, and vector of featues to be considered.
 *
 * 1. It calculates the total (weighted or unweighted) number of
 * valid panels that satisfy the quotas for its given `activeFeat`.
 * 2. It builds an explicit, directed acyclic graph (`adj`) of the states of the DP.
 * This will be used to build (`acumAdj`), which is used duting the rejaction sampling
 * process.
 * 4. It can efficiently recalculate the total count and graph
 * edge weights given a new set of respondent weights (`weights`) using a tree DP.
 * 5. It can accept a `shared_ptr` to a "parent" `CountSets`
 * instance (which solved a prefix of its features) to use as a pruning heuristic
 * 
 * The process is split into two phases:
 * 1. State Building: `findStatesDP` explores the state space to build an explicit 
 * state DAG. 
 * 2. Weighted Counting: `computeWeightedComb` (via `reWgtDfs`) traverses the 
 * existing DAG to calculate the total weighted counts (`memo`) based on current
 * respondent weights.
 */
 class CountSets: public enable_shared_from_this<CountSets>{ 
	// This is important for the recursive declaration of pointers
	private:

		/**
		*	Panel Data
		*/

		/** Const reference to the panel data (quotas, respondents). 
		* Note, this data might be preprocessed.
		* The truthfull data is always stored at the panel class.
		*/
		const PanelData &data;
		
		/** Const reference to the weight vector for each respondent. */
		const vector<dType> &weights;
		
		/** The target panel size. */
		int panSize;
		
		/**
		*	DP information
		*/

		/** Weak pointer to a "parent" DP instance solving a prefix of `activeFeat`. 
		 * Used to run a state punning heuristic. 
		 */
		weak_ptr<CountSets> prefDP;
		
		/** The number of features in the parent DP (size of the prefix). -1 if no parent. */
		int prefCatSz = -1;


		/** The number of features active in this DP instance. */
		long long numActiveFeat;
		
		/** The indices of the features from the data being solved for in this DP. */
		const vector<int> activeFeat;

		
		/** The total number of unique feature-value tuples found among all respondents. */
		int numTuple;
		
		/** Stores respondent indices, grouped by their unique feature-value tuple. 
		 * `respPerTuple[tupleId]` is a vector of respondent IDs. 
		 */
		vector< vector<int> > respPerTuple;
		
		/** Stores the total number of respondents belonging to each tuple. `tupleSz[tupleId]` is the count. */
		vector<int> tupleSz;

		/** Stores the feature-value vector for each unique tuple ID. 
		 * `valsTuple[tupleId]` is a `vector<int>` of feature values. 
		 */
		vector<FeatVec> valsTuple;

		/**
		 * Memoization & State Mapping
		 */

		/**
		 * @brief Memoization arrays for the initial structure-building phase.
		 * - `computedState`: A visited flag. True if `findStatesDP` has already processed this state ID.
		 * - `feasibleState`: The result of the Boolean DP. True if this state ID is part of a 
		 * path that leads to a valid, full panel. Used to prune the graph `adj`.
		 */
		vector<bool> computedState, feasibleState;


		/**  Memoization table for `FeatureMask` objects. Maps a mask to a unique ID. */
		boost::unordered_flat_map<FeatureMask, uint32_t, boost::hash<FeatureMask>> mskToId[__maxFeatures + 1];
		/**  Reverse mapping from a mask ID back to the `FeatureMask` object. */
		vector<FeatureMask> idToMsk[__maxFeatures + 1];

		/**  Memoization table for DP results. 
		 * `memo[stateId]` stores the (weighted) count for that state. 
		 * -1 if not computed. 
		 */
		vector<dType> memo;
		/** Memoization table for `DPState` objects. Maps a state to a unique ID. */
		boost::unordered_flat_map<DPState, uint32_t, boost::hash<DPState>> stateToId;


		/** Precomputed polynomial coefficients for weighted combinations for each tuple. 
		 * `CombCoefMatrix[tupleId][k]` is the sum of weights of all k-combinations from that tuple. 
		 */
		vector< vector<dType> > CombCoefMatrix;

		/** Flag indicating if this instance is only for reweighting an existing graph. */
		bool reWgt = 0;

		/** @brief Mapping from feature-value vectors to unique IDs*/
		boost::container::flat_map<FeatVec, int> tupleToId;


	public:

		/**
		 * Adjacency list representation of the explicit DP state graph.
		 * `adj[stateId]` contains a vector of outgoing edges.
		 * - `get<0>`: The ID of the destination state;
		 * - `get<1>`: The number of respondents chosen in this transition (f).
		 */
		vector< vector< tuple<uint32_t, uint16_t> > > adj;

		/**
		 * Cumulative probability table for sampling.
		 * `acumAdj[stateId]` stores a vector of normalized cumulative weights
		 * for the outgoing edges from `adj[stateId]`, used for fast weighted random
		 * selection in `getNextStateIdx`.
		 */
		vector< vector< floatType > > acumAdj;
		
		/** Reverse mapping from a state ID back to the `DPState` object. */
		vector<DPState> idToState;

		/** Maps state id from a feature-value vector. */
		vector<uint16_t> idToTuple;

		/**
		 * @brief Constructs and runs the DP calculation (or reweighting).
		 *
		 * This constructor sets up the DP computation:
		 * 1. Validates parent DP (if provided) and reserves memory.
		 * 2. Groups all respondents by their unique feature-value tuples (`vTuple`, `respPerTuple`).
		 * 3. Precomputes the weighted combination coefficients (`CombCoefMatrix`) for each tuple.
		 * 4. Clears memoization tables.
		 * 5. If not reweighting, it creates the initial state (`iniState`) and calls `findStatesDP` to
		 * solve the counting problem and populate the memo/adj structures.
		 * 6. If reweighting, it calls `reweight` to re-use an existing graph structure.
		 *
		 * @param panelData Reference to the `PanelData` (quotas, respondents).
		 * @param panelWeights Reference to the vector of respondent weights.
		 * @param activeFeat Vector of feature indices to be included in this DP.
		 * @param parentDP Optional `shared_ptr` to a parent DP solving a prefix of `activeFeat`.
		 */
		CountSets(const PanelData& panelData,
			const vector<dType>& panelWeights,
			const vector<int>& activeFeat,
			shared_ptr<CountSets> parentDP = nullptr) :
			data(panelData),
			weights(panelWeights),
			activeFeat(activeFeat),
			prefDP(parentDP){ // Initializes the weak_ptr to an optional parent DP.


			panSize = data.getPanSize(); // For readbility
			numActiveFeat = activeFeat.size();

			if (numActiveFeat > __maxFeatures)
				throw std::runtime_error("Exceed maximum number of features");

			// If a parent DP is provided, verify its features are a prefix of the current features.
			if (auto parent = prefDP.lock()) {
				auto prefCat = parent->getActiveFeat();
				assert(prefCat.size() <= activeFeat.size());

				for (int i = 0; i < prefCat.size(); i++)
					assert(prefCat[i] == activeFeat[i]);

				prefCatSz = prefCat.size();

				// Using the number of revious states to forecast memory usage
				long long nZState = parent->getNonZeroStateCount() * data.getFeatSize()[numActiveFeat - 1];

				idToState.reserve(nZState);
				memo.reserve(nZState);
				adj.reserve(nZState);
			}


			// Ensure all feature indices are valid.
			for (const int &x : activeFeat)
				assert((0 <= x) and (x < data.getTotFeature()));


			/*
			 *	WARNING: For the DP to function correctly, respondents must be processed
			 *	in an order where the first feature in `cat` is lexicographically sorted.
			*/

			// Identify all unique combinations of feature values (tuples) present in the pool.
			set<FeatVec> realTuple;
			for (auto &resp : data.getRespondents()){
				FeatVec aux;
				for (const int &x : activeFeat)
					aux.push_back(resp[x]); // Each respondent's featue-value vector

				realTuple.insert(aux);
			}

			// Assign a unique ID to each tuple.
			numTuple = 0;
			valsTuple.clear();
			tupleToId.clear();
			tupleSz.clear();
			for (auto tup : realTuple){
				valsTuple.push_back(tup);
				tupleSz.push_back(0);
				tupleToId[tup] = numTuple++;
			}

			// Group Respondents by their assigned tuple ID.
			respPerTuple.resize(numTuple);
			for (auto &x : respPerTuple)
				x.clear();

			int cnt = 0;
			for (auto &resp : data.getRespondents()){
				FeatVec aux;
				for (const int &x : activeFeat)
					aux.push_back(resp[x]);

				respPerTuple[ tupleToId[aux] ].push_back(cnt);
				tupleSz[ tupleToId[aux] ]++;
				cnt++;
			}

			// Precompute weighted combination coefficients for each respondent tuple.
			this->computeCombCoefs();
			

			// Clear all state-tracking structures.
			for (int i = 0; i <= __maxFeatures; i++){
				mskToId[i].clear();
				mskToId[i].rehash(0);
			}
			stateToId.clear();
			idToState.clear();
			memo.clear();
			adj.clear();

			computedState.clear();
			feasibleState.clear();

			// Check if this instance is for reweighting or a new DP calculation.
			reWgt = prefCatSz == activeFeat.size();

			if (!reWgt){
				/*
					Note: The DP is also capable of reweighting,
					but rurring a tree DP is way faster.
				*/

				if (__verbose_mode){
					cerr << YELLOW "Counting for features: " << WHITE;
					for (const int &x : activeFeat)
						cerr << (char)(x + 'a') << ' ';
					cerr << endl;
				}

				/*
				 *	Initial state
				 */

				DPState iniState;
				iniState.tupleId = 0;
				iniState.v0 = 0;
				iniState.sz = 0;
				iniState.numFeat = numActiveFeat;

				// Notice that hashed masks are shifted (maskIds[0] is for feature activeFeat_[1])
				for (int i = 1; i < numActiveFeat; i++){
					FeatureMask zeroMask(i, data.getFeatSize()[activeFeat[i]]);
					iniState.maskIds[i - 1] = getIdFromMsk(zeroMask); // Initializes this zero mask
				}

				// Start the recursive DP calculation from the initial state.
				auto z = findStatesDP(getIdFromState(iniState));


				idToTuple.resize(idToState.size());
				for (int i = 0; i < idToState.size(); i++){
					idToTuple[i] = idToState[i].tupleId;
				}

				if(__verbose_mode){
					if (z > 0)
						cerr << YELLOW << "\tDP is feasible with: " << WHITE << idToState.size() << " states" << endl;
					else
						cerr << YELLOW << "\tDP is infeasible with " << WHITE << idToState.size() << " states" << endl;

					long long nZState = idToState.size() - getNonZeroStateCount();
					cerr << YELLOW << "\tFraction of zero states: " << WHITE << setprecision(3) << 100 * (((double) nZState) / idToState.size()) << '%' << endl;
				}	
			}
			else{
				if (__verbose_mode){
					cerr << YELLOW << "Reweighting for features: " << WHITE;
					for (const int &x : activeFeat)
						cerr << (char)(x + 'a') << ' ';
					cerr << endl;
				}

				if (auto parent = prefDP.lock()){
					this->adj = parent->adj;
					this->idToState = parent->idToState;
					this->feasibleState = parent->feasibleState;
					this->idToTuple = parent->idToTuple;
				}

				computeWeightedComb();
			}
		}

		/**
		 * @brief Recalculates the combination coefficient matrix based on current weights.
		 * Must be called whenever the respondent weights change.
		 */
		void computeCombCoefs() {
			CombCoefMatrix.clear();
			CombCoefMatrix.reserve(numTuple);
			
			for (int i = 0; i < numTuple; i++){
				vector<dType> auxWgt;
				auxWgt.reserve(respPerTuple[i].size());
				
				// CountSets holds a const ref to weights. 
				// When Panel updates the vector values, this loop sees the new values.
				for (int rId : respPerTuple[i])
					auxWgt.push_back(weights[rId]);

				CombCoefMatrix.push_back(combWeigts(auxWgt));
			}
		}



		/**
		 * @brief Counts the number of non-zero states in the memoization table.
		 * @return The total count of states with a result > 0.
		 */
		const long long getNonZeroStateCount(){
			long long nZState = 0;
			for (int i = 0; i < idToState.size(); i++)
				if (feasibleState[i] > 0)
					nZState++;
			return nZState;
		}

		/**
		 * @brief Calculates weighted combination coefficients.
		 *
		 * For a set of weights `w_1, ..., w_n`, this computes the coefficients
		 * `C_k` of the polynomial `P(x) = \prod_{i=1}^{n}(1 + w_i * x) = \sum C_k * x^k`.
		 * `C_k` represents the sum of all k-wise products of weights.
		 *
		 * @param w The vector of weights `{w_1, ..., w_n}` for a respondent tuple.
		 * @return A vector `C` where `C[k]` is the coefficient `C_k`.
		 */
		vector<dType> combWeigts(const vector<dType> &w){
			long long n = w.size();

			// Base case: 0 degree polynomial
			vector<dType> prevRow(n + 1, 0);
			prevRow[0] = 1;  // C_0 is always 1

			if (n == 0)
				return prevRow;

			vector<dType> curRow(n + 1, 0);

			// Iteratively build the polynomial coefficients
			for (int i = 1; i <= n; ++i) {
				curRow[0] = 1; // Coefficient of x^0 is always 1
				for (int k = 1; k <= i; ++k) {
					// P_i(x) = P_{i-1}(x) * (1 + w_{i-1} * x)
				   // C(i, k) = C(i-1, k) + w_{i-1} * C(i-1, k-1)
					curRow[k] = prevRow[k] + w[i - 1] * prevRow[k - 1];
				}
				prevRow = curRow; // Current row becomes previous for the next iteration
			}

			return prevRow; // Return the final, complete vector
		}

		/**
		 * @brief Gets the precomputed combination coefficient.
		 * @param tId The tuple ID.
		 * @param k The number of respondents to choose from the tuple.
		 * @return The weighted combination coefficient `C_k` for tuple `tId`.
		 */
		const inline dType combCoef(long long tId, long long k){ return CombCoefMatrix[tId][k];}

		


		/**
		 * @brief Checks if quotas for the first (primary) feature can still be met.
		 *
		 * This is a specific check used in base case of the DP. It verifies
		 * that the current count `frqV0` meets the minimum for value `v0`, and
		 * that all subsequent values (from `v0 + 1` to end) have a minimum quota of 0.
		 * Which implies that the current solution is feasible for V0.
		 *
		 * @param v0 The current value index for the primary feature.
		 * @param frqV0 The current count for value `v0`.
		 * @param feat0 The feature index of the primary feature (`activeFeat[0]`).
		 * @return `true` if quotas can still be met, `false` otherwise.
		 */
		bool validFirstFeature(const int v0, const int frqV0, const int feat0){
			if (frqV0 < data.getMinVal()[feat0][v0])
				return false;

			int def = 0;
			for (int i = v0 + 1; i < data.getFeatSize()[ feat0 ]; i++)
				def += data.getMinVal()[ feat0 ][ i ];
			
			return def == 0;
		}

		/**
		 * @brief Checks if a FeatureMask represents a valid (quota-satisfying) state.
		 * This is a wrapper function that calls the implmenetation in mask.h
		 * @param mask The `FeatureMask` to check.
		 * @param feat The feature index corresponding to the mask.
		 * @return `true` if `mask.isValid` passes, `false` otherwise.
		 */
		inline bool validFeatureState(const FeatureMask& mask, const int feat){
			return mask.isValid(data.getMinVal()[feat], data.getMaxVal()[feat]);
		}

		/**
		 * @brief Gets a unique ID for a `FeatureMask`, creating one if it doesn't exist.
		 * Dont interact with the stucture directly, use the funtion.
		 * @param msk The `FeatureMask` to memoize.
		 * @return The unique ID (index) for this mask.
		 */
		long long getIdFromMsk(const FeatureMask &msk){
			int featId = msk.getFeatureId();

			if (mskToId[featId].count(msk))
				return mskToId[featId][msk];

			mskToId[featId][msk] = idToMsk[featId].size();
			idToMsk[featId].push_back(msk);

			return idToMsk[featId].size() - 1;
		}

		/**
		 * @brief Retrieves a `FeatureMask` object from its unique ID.
		 * @param mskId The ID of the mask.
		 * @return The corresponding `FeatureMask` object.
		 */
		inline FeatureMask getMskFromId(const int featId, const long long &mskId){
			return idToMsk[featId][mskId];
		}

		/**
		 * @brief Gets a unique ID for a `DPState`, creating one if it doesn't exist.
		 *
		 * This function is the gateway for creating new nodes in the DP graph.
		 * When a new, unseen state is encountered, it:
		 * 1. Creates a new entry in the adjacency list `adj`.
		 * 2. Creates a new entry in the memoization table `memo` (initialized to -1).
		 * 3. Assigns a new ID (the current size of `idToState`).
		 * 4. Stores the state object in `idToState`.
		 *
		 * @param state The `DPState` to memoize.
		 * @return The unique ID (index) for this state.
		 */
		long long getIdFromState(const DPState &state, const bool lookup = false){
			if (stateToId.count(state))
				return stateToId[state];

			// Exceeded the maximum number of states
			if (idToState.size() == std::numeric_limits<uint32_t>::max() - 1)
				throw DpStateException();

			if (lookup == false)
				adj.push_back({}); // Create a node in the state graph
			
			computedState.push_back(0);
			feasibleState.push_back(0);

			stateToId[state] = idToState.size();
			idToState.push_back(state);

			return idToState.size() - 1;
		}

		/**
		 * @brief Retrieves a `DPState` object from its unique ID.
		 * @param sId The ID of the state.
		 * @return The corresponding `DPState` object.
		 */
		inline DPState getStateFromId(const long long &sId){
			return idToState[sId];
		}


		/**
		 * The implementation of the dynamic programming counting algorithm
		 * 
		 * The DP calculates the (weighted) number of ways to select a quota-compliant
		 * panel of size `panSize`
		 * To do this, this DP calculates the number of ways to build a valid "panel profile".
		 *
		 * A "panel profile" is the final set of aggregate counts for all feature-value
		 * vectors (e.g., `{FeatA=1, FeatB=2}`). 
		 * 
		 * The DP iterates through each feature-value vector (from 0 to `numTuple-1`).
		 * At each `tupleId`, it counts: "How many (`f`) respondents
		 * this feature-value vector should be selected?"
		 *
		 * A `stateId` maps to a `DPState` object, which defines the subproblem:
		 * 
		 * "Given the current state, how many ways can we complete it to a valid panel?"
		 * 
		 * The total count is obtained from the empty state.
		 * 
		 * State Definition:
		 * 
		 *  - `tupleId`: The index of a feature-value vector `valsTuple[tupleId]`. 
		 *   We must decide how many respondents (`f`) to select with this vector.
		 *  - `sz`: The total number of respondents already selected from previous
		 *   feature-value vectors. This is derived from the masks.
		 *  - `v0`: The count of respondents selected so far that match the
		 *   CURRENT value of the primary feature (`activeFeat[0]`).
		 *  - `maskIds`: An array of IDs, where `maskIds[i-1]` maps to a
		 *   `FeatureMask` (`mskV[i]`) that stores the *full count distribution*
		 *   for feature `activeFeat[i]` (for all features `i > 0`).
		 *
		 * 
		 * Recurrence:
		 * 
		 *  Let `C(s)` be the (weighted) count of valid completions from a state `s`.
		 *  Let `s` be the state `(tupleId, v0, mskV[1...N-1])`.
		 *  Let `W(t, f)` be the weighted combination coefficient `combCoef(t, f)`,
		 *  which represents the sum of weights of all `f`-combinations of
		 *  respondents from tuple `t`.
		 *
		 *  The transition involves choosing `f` respondents from `tupleId`, which
		 *  leads to a new state `s'`.
		 *
		 *  `s'` = `(tupleId + 1, v0_new, mskV_new[1...N-1])`
		 *
		 *  The recurrence relation is:
		 *   `C(s) = Σ [ W(tupleId, f) * C(s') ]`
		 *  where the sum `Σ` is over all valid choices of `f` (from 0 to fmax).
		 *
		 *  `C(s)` is stored in (`memo[stateId]`).
		 * 
		 * 
		 *	The implementation of the DP will be divided in two parts for
		 * 	better use of the incremental approach for the features.
		 * 	
		 * 	- First, the DP finds the state graph, without computing the weighted 
		 * 	sums, this is done by `findStatesDP`
		 *  - Second, using the state graph, the DP recomptues  to compute the weighted 
		 * sums  according to the recurrence above. Given this, the reweighting
		 * 	step is natural. This is done by `computeWeightedComb`
		 */


		/**
		 * @brief The implementation of the structural part of dynamic programming 
		 * 	counting algorithm
		 * 
		 * @details
		 * This function calculates the state graph according the the recurrence above.
		 * 
		 * Feasibility Recurrence:
		 * 
		 *  Let `s` be the state `(tupleId, v0, mskV[1...N-1])`.
		 *  Let `F(s)` if state `s` has a valid completion.
		 *  The transition involves choosing `f` respondents from `tupleId`, which
		 *  leads to a new state `s'`.
		 *
		 *  `s'` = `(tupleId + 1, v0_new, mskV_new[1...N-1])`
		 *
		 *  The recurrence relation is:
		 *   `F(s) = [Σ [ F(s') ] > 0]`
		 *  where the sum `Σ` is over all valid choices of `f` (from 0 to fmax).
		 * 	That is, the recursion that if there is any feasible completion to a valid panel.
		 *
		 *  `F(s)` is stored in (`FeasibleState[stateId]`).
		 *
		 * 
		 * 
		 * Invariants:
		 * 
		 *  1. The `valsTuple` array is pre-sorted lexicographically, with 
		 *  `activeFeat[0]` as the primary sort key. This allows us to keep
		 *  track of que quotas for `activeFeat[0]` greedly (i.e., without 
		 *  using a mask) since we process all values within `activeFeat[0]`
		 *  sequentially.
		 *  - When the value for `activeFeat[0]` changes, the DP moves from 
		 *   a tuple with `vTuple[t][0] = 'A1'` to a tuple with `vTuple[t+1][0] = 'A2'`, 
		 *   it performs a quota check to guarantee that the current value of 
		 *   `activeFeat[0]` is satisfied.
		 *  - This is the biggest speedup in the DP, as we do not need to store a
		 *   `FeatureMask` for `activeFeat[0]`.
		 *  - All other features (`activeFeat[1...N-1]`) are NOT sorted, so
		 *   the counts for each value are stored in `FeatureMask`
		 *   objects (`mskV`) and carried through the recursion.
		 * 
		 *  2. The upper bounds of every value are always satisfied.
		 *
		 *
		 * 
		 * Base Cases:
		 * 
		 *  1. Panel is full (`sz == panSize`):
		 *   - The number of pool members selected meets the target panel size.
		 *    We must perform a final quota check.
		 * 		- `validFirstFeature()`: This verifies the quotas for the 
		 * 	 	 current `v0` value AND allc*subsequent* `v0` values (the DP
		 * 		 assumes that previous valus of `activeFeat[0]` are satisfied,
		 * 		 see Invariants).
		 * 		- `validFeatureState(mskV[i])` for `i > 0`: This checks the
		 * 		 the counts for the values within each feature.
		 *   - If valid: Return 1 (a single valid completion).
		 *   - Otherwise: Return 0 (invalid path).
		 *
		 *  2. Our of feature-vectors (`tupleId == numTuple && sz < panSize`):
		 *   - We have considered every feature-vector, but the panel is
		 *    still not full. Did not select enough.
		 *   - Return 0.
		 *
		 * 
		 * 
		 * Transition:
		 * 
		 *  The recursion is implemented as follows:
		 *  1. Find `maxSz`: Calculate the max number `f` we can select
		 *   from `tupleId` (current feature-vector). This is the minimum of:
		 * 	 - `panSize - sz` (slots Left)
		 *   - `tupleSz[tupleId]` (respondents available with this feature vector)
		 *   - `data.getMaxVal()[activeFeat[i]][featVal[i]] - freqVal[i]` 
		 * 	   (`maxQuota[v] - count[v]` for all features/values in this tuple, i.e,
		 * 	   upper quota capacity. This imposes that the upper-bound of every value is
		 * 	   satisfied, see Invariants).
		 * 
		 *  2. For `f` = `maxSz`..0:
		 *   - `comb = combCoef(tupleId, f)`: Get the weighted ways to choose `f` 
		 * 	  respondents with the current feature-value vector.
		 *   - Build `nxtState`: 
		 * 		-`tupleId + 1` 
		 * 		- Update `v0` by adding `f` 
		 * 		- Update the counts by adding `f` to the count of each value in 
		 * 		  valsTuple[tupleId]
		 *   - Check the quota for `v0`
		 * 		- If the next feature-value vector has a different value for
		 * 		  for `activeFeature[0]`, then, the quotas for the current value
		 * 		  need to be satisfied (this enforces the first invariant, see Invariants)
		 * 		- Since upper bounds are always satisfied (Invariant 2), we simply 
		 * 		  check the minQuota.
		 *   - If the next tuple has a different value for `activeFeature[0]`, we set `v0` to 0.
		 * 
		 *   - for each `f` we compute:
		 * 		`aux = findStatesDP(getIdFromState(nxtState))`
		 * 		If `aux > 0`, this is a valid path. 
		 * 		We store the transition `(nxtStateId, f)` in `adj`. 
		 * 		(note that, given this information, we can recompute the edge weights on-the-fly)
		 * 		`dpv |= aux` (We add if the state is feasible)
		 * 
		 * 
		 *  
		 * Heuristics and Pruning
		 *
		 *  1. Parent DP (`prefDP`) Lookup:
		 * 	 - If the current DP is considering the features `{h, g, a}` 
		 * 	  and `prefDP` considered features `{h, g}`, we can check the 
		 * 	  parent's solution to prune states.
		 * 
		 * 	 - We construct a "prefix state" `s_p` (containingconly the state 
		 *    for featues `{h, g}`) and call `parent->stateLookup(s_p)`.
		 * 	  If `stateLookup` returns 0, it means the subproblem
		 * 	  is already infeasible with fewer feature constraints. Thus,
		 * 	  it will be infeasible more constraints.
		 * 	  In practice, most of the states are infeasible, thus, this prunning
		 *    is very effective.
		 * 
		 *
		 *  2. Deficiency/Slack Pruning:
		 *   - We check if it's possible to complete the panel given the remaining 
		 * 	  slots and quotas.
		 * 	  Let `slotsLeft = panSize_ - sz`.
		 *   - Deficiency (`def`): The minimum number of pool members we
		 * 	  MUST add to meet all minimum quotas.
		 *   - Slack (`slk`): The maximum number of pool members we
		 *    CAN still add without violating any maximum quotas.
		 *   - If `slotsLeft < def`, we can't satisfy the minimums.
		 *   - If `slotsLeft > slk`, we can't satisfy the maximums.
		 * 
		 * 
		 *
		 * @param stateId The ID of the current DP state (node) to be computed.
		 * @param lookup A flag that indicates this is a call of the lookup heuristic.
		 * @return A boolean indicating that the current state is feasible.
		 * @throw DpTimeoutException If the global `__stopDPTimeout` flag is set by
		 * the watchdog thread.
		 */
		bool findStatesDP(long long stateId, bool lookup = 0){

			/*
				This flag is set to true by an auxiliary function with the DP times out
				We stop the DP by throwing an exception
			*/
			if (__stopDPTimeout)
				throw DpTimeoutException();
			

			if (computedState[stateId]) // Computed this state already
				return feasibleState[stateId];

			computedState[stateId] = 1;
			feasibleState[stateId] = 0;

			/*
				Recover the state
			*/

			DPState state = getStateFromId(stateId);

			auto tupleId = state.tupleId;
			int frqV0 = state.v0;
			int sz = state.sz;
			int numActiveFeat = state.numFeat;

			vector<FeatureMask> mskV(numActiveFeat); // mskV[0] is not used.
			for (int i = 1; i < numActiveFeat; i++)
				mskV[i] = getMskFromId(i, state.maskIds[i - 1]);
			
			if (numActiveFeat > 1)
				assert(sz == mskV[1].getSz()); // Current number of people in the partial panel.

			// assert(sz <= panSize); // Sanity check
			// for (int i = 1; i < numActiveFeat; i++)
			//     assert(sz == mskV[i].getSz());
			

			/*
				Base Case
			*/

			// Invalid state: did not meet panel size.
			if (tupleId == numTuple and sz < panSize){
				feasibleState[stateId] = 0;
				return 0;
			}

			FeatVec &featVal = valsTuple[tupleId]; // Values for the current tuple.

			// Panel is full. Check if quotas are met.
			if (sz == panSize){

				if (tupleId < numTuple){ // Check first feature
					if (!validFirstFeature(featVal[0], frqV0, activeFeat[0])){
						feasibleState[stateId] = 0;
						return 0;
					}
				}

				for (int i = 1; i < numActiveFeat; i++) // Check counts for all other features
					if (!validFeatureState(mskV[i], activeFeat[i])){
						feasibleState[stateId] = 0;
						return 0;
					}
				feasibleState[stateId] = 1;
				return 1;
			}
			

			vector<int> freqVal(numActiveFeat, 0);
			freqVal[0] = frqV0;
			for (int i = 1; i < numActiveFeat; i++)
				freqVal[i] = mskV[i].getCount(featVal[i]);


			/*
				Heuristics and Prunning
			*/

			
			/*
				Look-up heuristic
				Check if the state restricted to a prefix of features is feasible
				If not, the corrent one also isnt
			*/
			if (lookup == false and numActiveFeat > 2)
				if (auto parent = prefDP.lock()){
					FeatVec prefVal = featVal;
					vector<FeatureMask> prefMsk = mskV;
					while (prefVal.size() > prefCatSz){
						prefVal.pop_back();
						prefMsk.pop_back();
					}

					bool prefMemo = parent->stateLookup(prefVal, frqV0, sz, prefMsk);
					if (prefMemo == 0){
						feasibleState[stateId] = 0;
						return 0;
					}
				}

			// Heuristic: Check if remaining quotas (deficiency/slack) can be met.
			// Deficiency for the first feature.
			int defV0 = max<int>(0, data.getMinVal()[activeFeat[0]][featVal[0]] - frqV0);
			int slkV0 = data.getMaxVal()[activeFeat[0]][featVal[0]] - frqV0;
			for (int j = featVal[0] + 1; j < data.getFeatSize()[activeFeat[0]]; j++){
				defV0 += data.getMinVal()[activeFeat[0]][j];
				slkV0 += data.getMaxVal()[activeFeat[0]][j];
			}
			if (panSize - sz < defV0 || panSize - sz > slkV0){
				feasibleState[stateId] = 0; // Impossible to satisfy the first feature.
				return 0;
			}

			for (int i = 1; i < numActiveFeat; i++){
				long long def = mskV[i].getDeficiency(data.getMinVal()[activeFeat[i]]);
				if (panSize - sz < def){ // Now enough to satisfy the lower bounds
					feasibleState[stateId] = 0;
					return 0;
				}

				long long slack = mskV[i].getSlack(data.getMaxVal()[activeFeat[i]]);
				if (panSize - sz > slack){ // not enough to satisfy the upper bounds
					feasibleState[stateId] = 0;
					return 0;
				}
			}


	
			/*
				DP Transition
			*/


			// Determine the number of respondents `f` to select from the current tuple.
			// This is constrained by panel size, tuple size, and quotas.
			long long maxSz = min<long long>(panSize - sz, tupleSz[tupleId]);

			for (int i = 0; i < numActiveFeat; i++)
				maxSz = min<long long>(maxSz, data.getMaxVal()[activeFeat[i]][featVal[i]] - freqVal[i]);
						
			bool dpv = 0; // Total count for this state.

			DPState nxtState;
			nxtState.numFeat = numActiveFeat;

			// Iterate over all possible numbers of respondents `f` to take from this tuple.
			for (long long f = maxSz; f >= 0; f--){
				bool aux = 0;

				// Construct the next state.
				nxtState.tupleId = tupleId + 1;
				nxtState.v0 = freqVal[0] + f;
				nxtState.sz = sz + f;

				for (int i = 1; i < numActiveFeat; i++){
					FeatureMask nxtMsk = mskV[i]; // Copy current mask
					nxtMsk.incrementValue(featVal[i], f); // Increment the count of value featVal[i] by f
					nxtState.maskIds[i - 1] = getIdFromMsk(nxtMsk);
				}


				// Recurse, adjusting the next state based on the features of the next tuple.
				if (tupleId == numTuple - 1){ // Last tuple.
					if (freqVal[0] + f < data.getMinVal()[activeFeat[0]][featVal[0]])
						continue; // Lower bound for v0 must be met.
					
					aux = findStatesDP( getIdFromState(nxtState, lookup), lookup);
					
					if (aux > 0 and lookup == false)
						adj[ stateId ].push_back({ getIdFromState(nxtState), f});
				}
				else if (valsTuple[tupleId + 1][0] != featVal[0]) { // Next tuple has a new primary feature value.
					if (freqVal[0] + f < data.getMinVal()[activeFeat[0]][featVal[0]])
						continue; // Lower bound for v0 must be met.
					
					nxtState.v0 = 0; // Reset count for the new primary feature value.
					

					aux = findStatesDP( getIdFromState(nxtState, lookup), lookup);
					
					if (aux > 0 and lookup == false)
						adj[ stateId ].push_back({ getIdFromState(nxtState), f});
				
				} else { // Continue with the same primary feature value.
					aux = findStatesDP( getIdFromState(nxtState, lookup), lookup);
					
					if (aux > 0 and lookup == false)
						adj[ stateId ].push_back({ getIdFromState(nxtState), f});
				}
			
				dpv |= aux;
			}


			feasibleState[stateId] = dpv;
			return dpv;
		}


		/**
		 * @brief Heuristic check used to query a (parent) DP.
		 *
		 * Checks if a state restricted to this DP's feature prefix is feasible
		 * (i.e., has a non-zero count). This function may create new states
		 * in the parent DP if the quried state wasn't visited during the
		 * parent's own calculation.
		 *
		 * @param featVal The feature value vector for the state to check.
		 * @param frqV0 The primary feature count (`v0`) for the state.
		 * @param mskV The vector of `FeatureMask` objects for the state.
		 * @return `true` if the state has a count > 0, `false` otherwise.
		 */
		bool stateLookup(FeatVec &featVal, int frqV0, int sz, vector<FeatureMask> &mskV){

			int tId = tupleToId[featVal];

			/*
				Note: This might create a new mask
			*/
			DPState state;
			state.tupleId = tId;
			state.v0 = frqV0;
			state.sz = sz;
			state.numFeat = mskV.size();
			for (int i = 1; i < mskV.size(); i++){
				state.maskIds[i - 1] = getIdFromMsk(mskV[i]);
			}

			// if (stateToId.count(state) > 0)
			// 	return memo[stateToId.count(state)] > 0;

			// return 1;

			return findStatesDP(getIdFromState(state, true), true) > 0;
		}		

		

		/**
		 * @brief Generates a topological ordering of the state graph.
		 *
		 * @details
		 * This function exploits the strictly layered structure of the DP graph to perform
		 * a fast topological sort without traversing edges O(number of states).
		 *
		 * In this DP, transitions always move from a state with `tupleId = k` to a
		 * state with `tupleId = k + 1`. Therefore, the graph is a DAG layered by `tupleId`.
		 *
		 * @return A vector of state IDs in reverse topological order.
		 */
		vector<long long> topoSort;
		void findTopoSort(){
			long long numStates = adj.size();
			
			// Create buckets for each tuple layer
			// numTuple is the max, +1 for or end states
			vector<vector<long long>> layers(numTuple + 1); 


			long long nZState = 0;
			for (long long stateId = 0; stateId < numStates; stateId++) {

				int tupleId = (!idToTuple.empty()) ? idToTuple[stateId] : idToState[stateId].tupleId;

				
				if (tupleId < layers.size() and feasibleState[stateId]) {
					layers[tupleId].push_back(stateId);
					nZState++;
				}
			}

			topoSort.reserve(nZState);

			for (int t = 0; t <= numTuple; t++) {
				// Append all states in this layer to the result
				for (auto stateId : layers[t])
					topoSort.push_back(stateId);
			}
		}


		/**
		* @brief The implementation of the dynamic programming reweighting algorithm
		*
		* @details
		* This function updates the `memo` table to reflect changes 
		* in respondent weights without rebuilding the graph structure. 
		*  Note that, under the assumption that the weights are positive,
		*  the set of states and the transitions (stored in `adj`)
		*  depend only on the structure of the problem: the set of unique
		*  respondent tuples (`vTuple`) and the quotas (`data`).
		*  Changing respondent `weights` (as long as they remain positive) 
		*  only changes the value (the contribution)
		*  of a transition, not its existence. This function re-uses the existing,
		*  valid graph structure and simply re-calculates the
		*  values (weighted counts) for all states.
		* 
		* Process:
		* - Clears `acumAdj` to ensure sampling probabilities are strictly 
		* 	derived from the new weights.
		* - Calls `computeCombCoefs()` to recalculate weights ($C_k$) based 
		* 	on the new global weight vector.
		* - Traverses states in reverse topological order
		* -  If a state is a leaf (no edges), `memo[s]` is set to its `feasibleState` (1 if valid, 0 otherwise).
		* -  `memo[s] = Σ (CombCoef(tuple, f) * memo[next_state])` for all outgoing edges.
		*/
		void computeWeightedComb(){
			//Invalidate sampling structure
			acumAdj.clear();

			// Update combination coefficients based on new weights
			computeCombCoefs();


			long long numStates = adj.size();

			memo.assign(numStates, 0); 

			if (topoSort.empty())
				findTopoSort();

			for (int i = topoSort.size() - 1; i >= 0; i--){
				int stateId = topoSort[i];

				if (adj[stateId].empty())
					memo[stateId] = (int) feasibleState[stateId];

				int tupleId = (!idToTuple.empty()) ? idToTuple[stateId] : idToState[stateId].tupleId;

				for (auto &[nxtId, f] : adj[ stateId ]){
					dType comb = combCoef(tupleId, f);
					
					memo[stateId] += comb * memo[nxtId];
				}
			}
		}

		/**
		* @brief Prepares the DP state graph for efficient weighted sampling.
		*
		* This function creates the 'acumAdj' structure
		* `acumAdj[stateId]` is a vector of `floatType` values representing
		* the CDF. For example, if `adj_[s]` has edges with raw weights
		* `[10, 30, 60]`, `acumAdj_[s]` will store `[0.1, 0.4, 1.0]`.
		*
		* This pre-computation allows the `getNextStateIdx` function to
		* perform a fast, weighted random selection using just one random
		* number and a binary search (O(log k)).
		*
		* @tparam useFractional If true, uses high-precision `cpp_rational`
		* for normalization (slower, more accurate). If false, uses fast
		* `floatType` (less precise).
		*/
		template <bool useFractional>
		void buildNormalizedAdj() {
			if (!acumAdj.empty())
				return;

			try {
				acumAdj.resize(adj.size());
			} catch (const std::bad_alloc& e) {
				acumAdj.clear(); // Ensure clean state
				acumAdj.shrink_to_fit();
				throw std::runtime_error("Out of Memory: Failed to allocate accumulation table for sampling.");
			}
			
			// Atomic flag to signal OOM in a thread
			atomic<bool> rTErr(false);

			#pragma omp parallel for num_threads(__num_threads)
			for (size_t stateId = 0; stateId < adj.size(); stateId++){
				
				// Fast exit if another thread failed
				if (rTErr) continue;

				if (adj[stateId].empty())
					continue;
				
				try {
					auto &tupleId = idToTuple[stateId]; 

					// Use a local vector. This can throw OOM if huge.
					vector< tuple<dType, long long, int> > edges; 
					edges.reserve(adj[stateId].size());

					for (auto [nextId, f] : adj[stateId]){
						dType valComb = combCoef(tupleId, f) * memo[nextId];
						edges.emplace_back(valComb, nextId, f);
					}

					sort(edges.begin(), edges.end(), [](const auto& a, const auto& b){
						return get<0>(a) > get<0>(b);
					});

					dType sum = std::accumulate(edges.begin(), edges.end(), dType(0),
						[](dType curSum, const auto& edg) {
							return curSum + get<0>(edg);
						});

					if (sum == 0)
						continue;

					// Reserve inside inner vector
					acumAdj[stateId].reserve(adj[stateId].size());
					adj[stateId].clear(); 

					if constexpr (useFractional){
						dType acum = 0;
						for (const auto& [valComb, nextId, f] : edges){
							adj[stateId].push_back({nextId, f});
							acum += valComb;
							cpp_rational aux(acum, sum);
							acumAdj[stateId].push_back(aux.convert_to<floatType>());
						}
					} else {
						dType acum = 0;
						for (const auto& [valComb, nextId, f] : edges){
							adj[stateId].push_back({nextId, f});
							acum += valComb;
							floatType p = static_cast<floatType>(acum) / static_cast<floatType>(sum);

							acumAdj[stateId].push_back(p);
						}


					}

					if (!acumAdj[stateId].empty())
						acumAdj[stateId].back() = 1.0L;

				} catch (const std::exception& e) {
					rTErr = true;
				}
			}

			if (rTErr) {
				acumAdj.clear();
				acumAdj.shrink_to_fit();
				throw std::runtime_error("Exception while building accumulation table for sampling.");
			}
		}




		/**
		 * @brief Frees memory after the DP computation.
		 *
		 * Clears and releases memory from internal lookup tables
		 * (e.g., `idToMsk`, `mskToId`) and the `memo` cache. This function
		 * calls `shrink_to_fit()` and `rehash(0)` to reduce the object's
		 * memory footprint, retaining only data needed for reweighting
		 * and sampling.
		 */
		void clearMemory(){
			for (int i = 0; i <= __maxFeatures; i++){
				idToMsk[i].clear();
				idToMsk[i].shrink_to_fit();

				mskToId[i].clear();
				mskToId[i].rehash(0);
			}

			stateToId.clear();
			stateToId.rehash(0);

			if (idToTuple.empty()){
				idToTuple.resize(idToState.size());
				for (size_t i = 0; i < idToState.size(); i++){
					idToTuple[i] = idToState[i].tupleId;
				}
			}
			
			

			idToState.clear();
			idToState.shrink_to_fit();
		}


		/** @brief Gets the total (weighted) count of all valid panels (result at root node). */
		inline dType getAllSets(){ return memo[0];}
		/** @brief Gets the vector of active feature indices for this DP. */
		inline const vector<int> getActiveFeat(){ return activeFeat; }
		/** @brief Gets the map of respondents grouped by tuple ID. */
		inline const vector< vector<int> > getRespPerTuple(){ return respPerTuple; }
		/** @brief Gets a const reference to the PanelData object. */
		inline const PanelData& getPanelData() const { return data; }
};



/**
 * @class Panel
 * @brief High-level orchestrator for panel data loading, counting, and sampling.
 *
 * This class serves as the main user-facing API. It owns the `PanelData`
 * and manages a cache (`countSets_`) of `CountSets` DP solver instances
 * for different feature combinations. It provides public methods for
 * counting, sampling, reweighting, and automatic feature selection.
 */
class Panel {
	private:
		/** The target panel size. */
		int panSize;
		/** The original, unmodified panel data loaded from files. */
		PanelData data;
		/** @brief A copy of the panel data, potentially preprocessed (e.g., value-merged). */
		PanelData processedData;
		/** The current vector of weights for all respondents (defaults to 1). */
		vector<dType> weights;


		/** Cache of DP results. Maps a feature set (vector<int>) to its solver instance. */
		map<vector<int>, shared_ptr<CountSets>> countSets;
		/**  Cached pointer to the DP result from `autoFeatureSelection`. */
		shared_ptr<CountSets> autoSelectedCountSets;
		/** Cached pointer to a DP instance with modified weights. */
		shared_ptr<CountSets> reCountSets;
		
	public:

		/** Total number of samples/tries from the last `samplingAlgorithm` call. */
		long long __totalSample;

		/**
		 * @brief Constructs a Panel instance.
		 *
		 * Loads data from the specified CSV files into the `data` member and
		 * creates a copy in `processedData`. Initializes respondent weights to 1.
		 *
		 * @param panSize The target panel size.
		 * @param categoriesFile Path to the categories (features/quotas) CSV file.
		 * @param respondentsFile Path to the respondents CSV file.
		 */
		Panel(int panSize,
			const string &categoriesFile,
			const string &respondentsFile)
			: panSize(panSize), data(panSize, categoriesFile, respondentsFile), processedData(data){
			weights.resize(data.getRespondents().size(), 1);
		}

		
		/** @brief Clears all cached DP results. */
		void clear(){ countSets.clear(); }

		/**
		 * @brief Counts the valid panels for a given set of features.
		 *
		 * Uses a cache (`countSets`) to avoid re-computation.
		 * Can optionally build DPs progressively (e.g., solve {a,b} before {a,b,c})
		 * to use the smaller solution as a heuristic for the larger one.
		 *
		 * @param features A vector of feature indices to solve for.
		 * @param progressive If true, build DPs for prefixes first.
		 * @return The total (weighted) count of valid panels.
		 */
		dType count(const vector<int> &features, bool progressive = true) {
			auto it = countSets.find(features);
			if (it != countSets.end())
				return it->second->getAllSets();

			if (reCountSets.get() != nullptr){
				cerr << PURPLE << "Attention:" << WHITE << " counting non-standard weights needs to be done though reweighting. ";
			}

			if (features.empty())
				throw std::runtime_error("Cannot count an empty feature set.");

			// Base case: direct build
			if (!progressive || features.size() == 1) {
				auto [it, inserted] = countSets.try_emplace(
					features,
					make_shared<CountSets>(data, weights, features, nullptr)
				);

				it->second->computeWeightedComb();
				return it->second->getAllSets();
			}
			
			// Progressive case: start from the first singleton.
			vector<int> prefCat = {features[0]};
			shared_ptr<CountSets> prevPtr;
			
			auto [firstIt, _] = countSets.try_emplace(
				prefCat,
				make_shared<CountSets>(data, weights, prefCat, nullptr)
			);
			prevPtr = firstIt->second;

			while (prefCat.size() < features.size()) {
				prefCat.push_back(features[prefCat.size()]);
				auto [newIt, __] = countSets.try_emplace(
					prefCat,
					make_shared<CountSets>(data, weights, prefCat, prevPtr)
				);
				prevPtr = newIt->second;
			}

			prevPtr->computeWeightedComb();
			return prevPtr->getAllSets();
		}

		/*
			Sampling funtions
		*/


		/*
			Wrapper Functions
		*/

		/**
		 * @brief Samples a single panel uniformly (weights = 1).
		 * @param dp The computed `CountSets` DP object.
		 * @param maxSamples Safeguard limit on total rejection attempts.
		 * @return A vector of respondent IDs, or an empty vector if no panel is found.
		 */
		vector<int> sampleUniformPanel(const shared_ptr<CountSets> &dp, long long maxSamples = -1){
			// IsWeighted=false and pruneSearch=true.
			vector< vector<int> > panels = samplingAlgorithm<false, true>(dp, 1, maxSamples);
			
			if (panels.empty())
				return {};

			return panels[0];
		}

		/**
		 * @brief Samples a single panel uniformly (weights = 1) given a feature vector, 
		 * computing the DP if needed.
		 * @param features The feature set to use for sampling.
		 * @param maxSamples Safeguard limit on total rejection attempts.
		 * @return A vector of respondent IDs, or an empty vector.
		 * @throw std::runtime_error If the DP for `features` cannot be found or computed.
		 */
		vector<int> sampleUniformPanel(const vector<int> &features, long long maxSamples = -1) {
			// Ensure the DP graph is computed and get the shared_ptr.
			count(features);
			auto dp = countSets.find(features);

			// Handle case where features might be invalid.
			if (dp == countSets.end()) {
				throw std::runtime_error("Could not find or compute DP for the given features.");
			}

			return sampleUniformPanel(dp->second, maxSamples);
		}

		/**
		 * @brief Samples a single panel using weighted respondent selection.
		 * @param dp The computed `CountSets` DP object.
		 * @param maxSamples Safeguard limit on total rejection attempts.
		 * @return A vector of respondent IDs, or an empty vector if no panel is found.
		 */
		vector<int> sampleWeightedPanel(const shared_ptr<CountSets> &dp, long long maxSamples = -1){
			// IsWeighted=true and pruneSearch=true.
			vector< vector<int> > panels = samplingAlgorithm<true, true>(dp, 1, maxSamples);
			
			if (panels.empty())
				return {};

			return panels[0];
		}

		/**
		 * @brief Calculates respondent frequencies over N panels using uniform selection.
		 * @param dp The computed `CountSets` DP object.
		 * @param nPanels The target number of panels to sample.
		 * @param maxSamples Safeguard limit on total rejection attempts.
		 * @return A vector of frequencies, where `vec[i]` is the count for respondent `i`.
		 */
		vector<long long> getUniformFrequencies(const shared_ptr<CountSets> &dp, long nPanels, long long maxSamples = -1){
			// IsWeighted=false and pruneSearch=true.
			vector< vector<int> > panels = samplingAlgorithm<false, true>(dp, nPanels, maxSamples);

			vector<long long> respFreq(data.getRespondents().size(), 0);

			for (vector<int> pan : panels)
				for (int rId : pan){
					assert(rId < respFreq.size());
					respFreq[rId]++;
				}

			return respFreq;
		}

		/**
		 * @brief Calculates respondent frequencies (uniform) by feature vector, computing the DP if needed.
		 * @param features The feature set to use for sampling.
		 * @param nPanels The target number of panels to sample.
		 * @param maxSamples Safeguard limit on total rejection attempts.
		 * @return A vector of frequencies.
		 * @throw std::runtime_error If the DP for `features` cannot be found or computed.
		 */
		vector<long long> getUniformFrequencies(const vector<int> &features, long nPanels, long long maxSamples = -1){
			count(features);
			auto dp = countSets.find(features);

			// Handle case where features might be invalid.
			if (dp == countSets.end()) {
				throw std::runtime_error("Could not find or compute DP for the given features.");
			}
	
			return  getUniformFrequencies(dp->second, nPanels, maxSamples);
		}

		/**
		 * @brief Calculates respondent frequencies over N panels using weighted selection.
		 * @param dp The computed `CountSets` DP object.
		 * @param nPanels The target number of panels to sample.
		 * @param maxSamples Safeguard limit on total rejection attempts.
		 * @return A vector of frequencies.
		 */
		vector<long long> getWeightedFrequencies(shared_ptr<CountSets> &dp, long nPanels, long long maxSamples = -1){
			// IsWeighted=true and pruneSearch=true.
			vector< vector<int> > panels = samplingAlgorithm<true, true>(dp, nPanels, maxSamples);

			vector<long long> respFreq(data.getRespondents().size(), 0);

			for (auto pan : panels)
				for (auto rId : pan)
					respFreq[rId]++;


			return respFreq;
		}

		/*
			Sampling Algorithm
		*/


		/**
		 * @brief Selects the next state index during a weighted random walk on the DP graph.
		 *
		 * Uses the precomputed cumulative probability table `acumAdj` for the
		 * current `stateId`. Generates a random number in [0.0, 1.0) and uses
		 * `std::upper_bound` (binary search) to find the corresponding edge index.
		 *
		 * @param stateId The ID of the current DP state (graph node).
		 * @param rng A reference to the thread-local random number generator.
		 * @param it A `shared_ptr` to the `CountSets` object containing the graph.
		 * @return The index of the chosen outgoing edge in `it->adj_[stateId]`.
		 */
		inline long long getNextStateIdx(const long long stateId, std::mt19937_64& rng,
								const shared_ptr<CountSets>& it) {
			const auto& cumSum = it->acumAdj[stateId];
			if (cumSum.empty()) return -1;

			static thread_local std::uniform_real_distribution<floatType> dist(0.0, 1.0);
			floatType r = dist(rng);

			if (r < cumSum[0]) return 0;
			if (cumSum.size() == 1) return 0;

			if (r < cumSum[1]) return 1;
			if (cumSum.size() == 2) return 1;

			if (r < cumSum[2]) return 2;

			// Binary search (and clamp for safety)
			auto idx = (long long)distance( cumSum.begin(),upper_bound(cumSum.begin(), cumSum.end(), r));
			if (idx >= (long long)cumSum.size()) 
				idx = (long long)cumSum.size() - 1;
			return idx;
		}


		/**
		 * @brief Performs a partial Fisher-Yates shuffle uniformly.
		 * Selects `k` elements from the range `[first, last)` and places them
		 * at the beginning `[first, first + k)`.
		 *
		 * @tparam rIt Random access iterator type.
		 * @tparam RNG Random number generator type.
		 * @param first Iterator to the beginning of the range.
		 * @param last Iterator to the end of the range.
		 * @param k The number of elements to select.
		 * @param rng The random number generator.
		 */
		template<typename rIt, typename RNG>
		void partialUniformShuffle(const rIt first, const rIt last, size_t k, RNG& rng){
			size_t n = distance(first, last);
			k = min(k, n);
			
			for (size_t i = 0; i < k; i++){
				std::uniform_int_distribution<size_t> dist(i, n - 1);
				iter_swap(first + i, first + dist(rng));
			}
		}



		/**
		 * @brief Selects `k` elements from a range via a weighted partial shuffle.
		 *
		 * Performs an in-place weighted partial shuffle. The elements `[first, last)`
		 * are treated as *indices* into the `respMap`, which in turn maps to the
		 * global `weights` vector.
		 *
		 * @tparam rIt Random access iterator type.
		 * @tparam RNG Random number generator type.
		 * @param first Iterator to the beginning of the range of indices.
		 * @param last Iterator to the end of the range of indices.
		 * @param respMap A vector mapping local indices (from the tuple) to global respondent IDs.
		 * @param k The number of elements to select.
		 * @param rng The random number generator.
		 */
		template<typename rIt, typename RNG>
		void partialWeightedShuffle(const rIt first, const rIt last, const vector<int> &respMap, size_t k, RNG& rng){
			size_t n = distance(first, last);
			k = min(k, n);

			if (k == 0) return;

			/*
				Create a vector of weights corresponding to the iterators.
				respMap contains the actual ids of the respondents
				discrete_distribution does not take cpp_int
			*/
			vector<double> currWeights;
			currWeights.reserve(n);
			for (auto it = first; it != last; it++)
				currWeights.push_back(generic_convert<double>(weights[respMap[*it]]));
			
			for (size_t i = 0; i < k; i++){
				/*
					Created a weighted distribution
					It will return relative indices [0, ..., currWeights - i)
				*/
				std::discrete_distribution<size_t> dist(currWeights.begin() + i, currWeights.end());

				size_t idx = i + dist(rng);

				// Swap the chosen element into the current position 'i'
				iter_swap(first + i, first + idx);
				swap(currWeights[i], currWeights[idx]);
			}
		}
		


		/*
			Benchmarking/Diagnostics information
			Will always be associated to the lastest call of the function
		*/
		long long __number_of_tries; // total tries in last diagnostic/sampling run
		vector<long long> __feat_sat; // __feat_sat[f] = #tries feature f satisfied
		vector<vector<long long>> __val_sat; // __val_sat[f][v] = #tries value (f,v) satisfied


		/*
			This tracks the which features prune panels
		*/
		vector<long long> __panels_pruned;



		/** Next time a non-final sampling info print is allowed. */
		chrono::steady_clock::time_point nextSamplingInfoPrint;

		/**
		* @brief Implementation of the weighted sampling algorithm.
		* 
		* @details It samples panels according to a probability distribution 
		* defined by respondent weights, using the pre-computed DP state graph.
		* 
		* 
		* 
		* The panel distribution:
		* 
		* The weight of a panel P = \{r_1, ..., r_k\} is the product of its
		* members' weights: W(P) = w_1 X ... X w_k. The DP graph
		* (`memo[0]`) computes the sum of weights of all valid panels, W_{total}.
		* 
		* This algorithm samples a panel $P$ with probability P(P) = W(P) / W_{total}.
		* 
		*  - Weighted Sampling (`IsWeighted = true`): Uses the respondent
		* 	weights w_i as provided.
		*  - Uniform Sampling (`IsWeighted = false`): This is a special
		* 	case where all $w_i = 1$, resulting in a uniform probability
		*   distribution $P(P) = 1 / \text{TotalPanels}$.
		* 
		* 
		* 
		* Sampling Algorithm:
		* 
		* The algorithm generates a panel in 3 stages:
		* 
		*  1. Sample a "Panel Profile":
		* 	The DP graph `adj` (with its pre-computed CDF in `acumAdj`)
		* 	is used as a probabilistic model. A weighted random walk is
		* 	performed (using `getNextStateIdx`) to select a valid "profile"
		* 	(a set of counts `f` for each feature-value vector).
		* 
		*  2. Sample Respondents:
		* 	When `f` respondents from a tuple are selected, this
		* 	function calls either `partialWeightedShuffle` or
		* 	`partialUniformShuffle` to select the `f` respondents
		* 	with that feature-value vector, according to their weights.
		*  
		*  3. Rejaction
		* 	The candidate panel is validated against all quotas from
		* 	`data`. If it fails, it is rejected, and the process repeats.
		* 
		* 
		* @tparam IsWeighted If true, samples panels proportionally to their
		*  $W(P) = \prod w_i$. If false, samples uniformly (all $w_i=1$).
		* @tparam pruneSearch If true, validate against all quotas during
		*  the random walk and abort early if any are violated.
		* @param dp The computed `CountSets` DP object (must have graph `adj`
		*  built).
		* @param nPanels The target number of valid panels to find.
		* @param maxSamples Safeguard limit on total rejection attempts across
		*  all threads. If -1, no limit.
		* @return A `vector<vector<int>>` containing the valid panels found.
		*  Each inner `vector<int>` is a single panel composed of
		*  respondent IDs.
		*/
		template<bool IsWeighted, bool pruneSearch>
		vector< vector<int> > samplingAlgorithm(const shared_ptr<CountSets> &dp,
			const long nPanels,
			long long maxSamples = -1){

			/*
				Build the adjacency list for the sampling.
			*/

			try {
				if constexpr (std::numeric_limits<dType>::is_integer){
					dp->buildNormalizedAdj<true>(); // Normalization using integers
				} else {
					dp->buildNormalizedAdj<false>(); // Normalization using floating points
				}	
			} catch (const std::exception& e) {
				if (__verbose_mode || __test_mode) {
					cerr << RED << "Error during sampling: " << e.what() << WHITE << endl;
				}
				// Return empty to signal failure
				return {};
			}


			/**
				Clean all feature satisfaction probabilities 
			*/
			__number_of_tries = 0;
			__feat_sat.assign(data.getTotFeature(), 0);
			__val_sat.assign(data.getTotFeature(), {});
			for (int f = 0; f < data.getTotFeature(); f++) 
				__val_sat[f].assign(data.getFeatSize()[f], 0);

			__panels_pruned.assign(data.getTotFeature(), 0);


			bool abortSearch = false; // Flag to stop all threads

			const vector< vector<int> > respPerTuple = dp->getRespPerTuple();


			atomic<long long> nPanelsFound(0);
			vector< vector<int> > retPanels;


			nextSamplingInfoPrint =
				chrono::steady_clock::now() + chrono::seconds(__sampling_update_seconds);

			#pragma omp parallel num_threads(__num_threads) \
			shared(dp, __number_of_tries, respPerTuple, __totalSample, abortSearch)
			{

				/*
					Rejaction Sampling Loop
				*/


				// Each thread uses its own RNG, seeded once in main.cpp.
				std::mt19937_64 &rng = __thread_rngs[omp_get_thread_num()];

				// Only delcared once per thread
				vector<int> panel(panSize);

				vector< vector<int> > localPanels;

				int curPos = 0;

				vector< vector<int> > panValues(data.getTotFeature());
				for (int f = 0; f < data.getTotFeature(); f++)
					panValues[f].assign(data.getFeatSize()[f], 0);



				long long localTries = 0;
				vector<long long> localFeatSat(data.getTotFeature(), 0);

				vector<vector<long long>> localValSat(data.getTotFeature());
				for (int f = 0; f < data.getTotFeature(); f++) 
					localValSat[f].assign(data.getFeatSize()[f], 0);

				vector<long long> localPanelsPrunned(data.getTotFeature(), 0);



				// Vector to sample indexes from
				vector<int> indices(data.getRespondents().size());

				do{
					localTries++;
					if (localTries % __sampling_update_tries == 0){ // update global counters periodically

						updateSamplingStatistics(localTries, localPanelsPrunned);

						if (__test_mode || __panelot_output){
							#pragma omp critical
							{
								printSamplingInfo(dp, 0, 0);
							}
						}

						if (maxSamples > 0 and __number_of_tries > maxSamples) // Exceeded number of tries
							abortSearch = 1;
					}


					for (int f = 0; f < data.getTotFeature(); f++){
						fill(panValues[f].begin(), panValues[f].end(), 0);
					}
					
					curPos = 0;
				
					long long stateId = 0;
					bool feasiblePanel = 1;

					// Traverse the DP state graph to build a candidate panel.
					while (dp->adj[ stateId ].size() and feasiblePanel and !abortSearch){

						long long edgeId = getNextStateIdx(stateId, rng, dp);
	

						long long nxtId = get<0>(dp->adj[stateId][edgeId]);
						int f = get<1>(dp->adj[stateId][edgeId]);


						auto &tupleId = dp->idToTuple[stateId];

						const auto& tupleResps = respPerTuple[tupleId];
						int availableResps = tupleResps.size();
						if (availableResps > 0 && f > 0) {
							iota(indices.begin(), indices.begin() + availableResps, 0);
							

							/*
								Selects the appropriate subpanel sampling function
								depending on weighted or unwighted.
							*/
							
							if constexpr (IsWeighted){
								partialWeightedShuffle(indices.begin(), indices.begin() + availableResps, tupleResps, f, rng);
							} else {
								partialUniformShuffle(indices.begin(), indices.begin() + availableResps, f, rng);
							}


							// Select the f respondents using the shuffled positions
							for (int i = 0; i < f; i++){
								int rId = tupleResps[indices[i]];
							
								panel[curPos++] = rId;
							
								for (int j = 0; j < data.getTotFeature(); j++){
									int valueID = data.getRespondents()[rId][j];
									panValues[j][valueID]++;
									

									/*
										Enable early prunning
									*/
									if constexpr (pruneSearch){
										if (data.getMaxVal()[j][valueID] < panValues[j][valueID] and feasiblePanel){
											feasiblePanel = 0;
											localPanelsPrunned[j]++;
										}
									}
								}
							}
						}
					
						stateId = nxtId;
					}

					if (!feasiblePanel)
						continue;


					// Check if the generated panel satisfies all feature quotas.
					// Also computes feature and value satisfaction probabilities
					int numSatisfied = 0;
					for (int f = 0; f < data.getTotFeature(); f++){ // For each feature
						bool feature_sat = 1;

						for (int v = 0; v < data.getFeatSize()[f]; v++){ // For each value
							bool value_sat = 1;
							if ((panValues[f][v] < data.getMinVal()[f][v]) or (data.getMaxVal()[f][v] < panValues[f][v])){
								feature_sat = 0;
								value_sat = 0;
							}

							localValSat[f][v] += value_sat;
						}

						if (feature_sat == 0 and feasiblePanel == 1){
							localPanelsPrunned[f]++;
							feasiblePanel = 0;
						}


						numSatisfied += feature_sat;
						localFeatSat[f] += feature_sat;
					}

					if (numSatisfied < data.getTotFeature() or curPos < data.getPanSize())
						feasiblePanel = 0;


					if (feasiblePanel){
						
						int curTotPanels = nPanelsFound.fetch_add(1, std::memory_order_relaxed);
						if (curTotPanels < nPanels){

							localPanels.push_back(panel);
							
							updateSamplingStatistics(localTries, localPanelsPrunned);

							#pragma omp critical
							{
								printPanel(panel);
								if (__panelot_output)
									printSamplingInfo(dp, 0, 1);
							}
								
							if (curTotPanels == nPanels - 1)
								abortSearch = true;
						}
					}
						
				} while (!abortSearch);
			

				/**
					Merge all satisfcation probabilities measures
				*/
				#pragma omp critical
				{
					updateSamplingStatistics(localTries, localPanelsPrunned);

					for (int f = 0; f < data.getTotFeature(); f++){
						__feat_sat[f] += localFeatSat[f];
						for (int v = 0; v < data.getFeatSize()[f]; v++)
							__val_sat[f][v] += localValSat[f][v];
					}

					while (!localPanels.empty()){
						retPanels.push_back(localPanels.back());
						localPanels.pop_back();
					}
				}
			}



			if (__test_mode)
				printSamplingInfo(dp, 1, 1);

			return retPanels;
		}


		void updateSamplingStatistics(long long &localTries, vector<long long> &localPanelsPrunned){
			#pragma omp atomic update
				__number_of_tries += localTries;

			localTries = 0;	

			for (int f = 0; f < data.getTotFeature(); f++){
				#pragma omp atomic update
					__panels_pruned[f] += localPanelsPrunned[f];
				localPanelsPrunned[f] = 0;	
			}
		}





		/**
		 * @brief Applies new weights and re-computes the DP solution.
		 *
		 * Optimizes performance by modifying the existing reCountSets object 
		 * in-place if it exists and matches the requested features.
		 *
		 * @param features The feature set to reweight.
		 * @param newWeights The new vector of weights for all respondents.
		 * @return The total (newly weighted) count of valid panels.
		 */
		dType reweight(const vector<int> &features, const vector<dType> &newWeights){
			// Ensure the base DP exists
			count(features);

			// Update the weights vector. 
			// Since CountSets holds a reference to this vector, it will see the changes.
			assert(weights.size() == newWeights.size());
			for (int i = 0; i < data.getRespondents().size(); i++)
				weights[i] = newWeights[i];


			if (reCountSets != nullptr) {
				// Verify we are reweighting the same feature set
				if (reCountSets->getActiveFeat() == features) {
					reCountSets->computeWeightedComb();
					return reCountSets->getAllSets();
				}
			}

			// Fallback: Create new reweighted set (first time or different features)
			reCountSets = make_shared<CountSets>(data, weights, features, countSets[features]);

			return reCountSets->getAllSets();
		}


		/**
		 * @brief Applies new weights and re-computes DP
		 * @param dp A `shared_ptr` to the `CountSets` instance to use as a basis.
		 * @param newWeights The new vector of weights for all respondents.
		 * @return A `shared_ptr` to the reweighted `CountSets` instance (same object as dp).
		 */
		shared_ptr<CountSets> reweight(const shared_ptr<CountSets> &dp, const vector<dType> &newWeights){
			// 1. Update the Panel's master weights vector.
			// Since 'dp' was constructed with a reference to 'this->weights', 
			// it will see these new values when it recomputes.
			assert(weights.size() == newWeights.size());
			weights = newWeights;

			reCountSets = dp;

			reCountSets->computeWeightedComb();

			return reCountSets;
		}

		/** @brief Resets all respondent weights to 1 and clears the reweight cache. */
		void resetWeights(){
			fill(weights.begin(), weights.end(), 1);
			reCountSets = shared_ptr<CountSets>();
		}

		/*
			Autoselection and Preprocessing
		*/



		void printSelectionProbTable(const string& title,
							const vector<int>* activeFeat = nullptr) const {

			const int m = data.getTotFeature();

			cerr << YELLOW << "\n--- " << title << " ---" << WHITE << endl;
			cerr << YELLOW << "Tries: " << WHITE << __number_of_tries << endl;

			vector<char> isActive(m, 0);
			if (activeFeat) {
				for (int f : *activeFeat) isActive[f] = 1;
			}

			int maxV = 0;
			for (int f = 0; f < m; f++) maxV = max(maxV, data.getFeatSize()[f]);

			const int colW = 8;

			cerr << setw(colW) << "Feature" << " |"
				 << setw(colW) << "P(feat)";
			for (int v = 0; v < maxV; v++)
				cerr << setw(colW) << ("Val " + to_string(v));
			cerr << endl;

			cerr << string(colW + 2 + colW * (1 + maxV), '-') << endl;

			for (int f = 0; f < m; f++) {
				string name = string(" ") + (char)('a' + f);
				if (isActive[f]) name += "*";

				cerr << setw(colW) << name << " |";

				long double pFeat = (long double)__feat_sat[f] / (long double)__number_of_tries;
				cerr << fixed << setprecision(4) << setw(colW) << (double)pFeat;

				for (int v = 0; v < data.getFeatSize()[f]; v++) {
					long double pVal = (long double)__val_sat[f][v] / (long double)__number_of_tries;
					cerr << fixed << setprecision(4) << setw(colW) << (double)pVal;
				}
				// pad missing columns
				for (int v = data.getFeatSize()[f]; v < maxV; v++)
					cerr << setw(colW) << "";

				cerr << endl;
			}
			cerr << YELLOW << "-----------------------------" << WHITE << endl << endl;
		}






		/**
		 * @brief Populates `__value_freq` with feature satisfaction data from uniform sampling.
		 *
		 * Simulates `numSamples` uniform random panels and records the frequency
		 * of each count (0 to panSize) for each feature value. This data is
		 * used by `getFeatureSatisfactionProb`.
		 *
		 * @param currentData The `PanelData` (respondents, quotas) to sample against.
		 * @note Modifies `__value_freq` and `__number_of_tries` members.
		 */
		void runUniformDiagnostics(long long numSamples = 1'000'000){
			if (__test_mode)
				cerr << BLUE << "Satisf Prob uniform k-set: " << WHITE << numSamples << endl;

			const int m = data.getTotFeature();
			const long long n = (long long)data.getRespondents().size();

			vector<int> respIdx(n);
			iota(respIdx.begin(), respIdx.end(), 0);

			std::mt19937_64 rng((uint64_t)chrono::steady_clock::now().time_since_epoch().count());

			__number_of_tries = 0;
			__feat_sat.assign(m, 0);
			__val_sat.assign(m, {});
			for (int f = 0; f < m; f++)
				__val_sat[f].assign(data.getFeatSize()[f], 0);

			vector<vector<int>> panValues(m); // Frequency of each value in the panel
			for (int f = 0; f < m; f++)
				panValues[f].assign(data.getFeatSize()[f], 0);

			for (long long s = 0; s < numSamples; s++) {
				shuffle(respIdx.begin(), respIdx.end(), rng); // Random set of respondents

				for (int f = 0; f < m; f++)
					fill(panValues[f].begin(), panValues[f].end(), 0);

				for (int k = 0; k < panSize; k++) {
					int rId = respIdx[k];
					const auto& respondent = data.getRespondents()[rId];
					for (int f = 0; f < m; f++) {
						int val = respondent[f];
						panValues[f][val]++;
					}
				}

				for (int f = 0; f < m; f++) {
					bool feat_sat = true;

					for (int v = 0; v < data.getFeatSize()[f]; v++) {
						bool value_sat = (data.getMinVal()[f][v] <= panValues[f][v] && panValues[f][v] <= data.getMaxVal()[f][v]);
						__val_sat[f][v] += (long long) value_sat;
						feat_sat &= value_sat;
					}

					__feat_sat[f] += (long long)feat_sat;
				}

				__number_of_tries++;
			}
		}

		

		vector<int> chooseInitialOrder(const PanelData& processedData) const {
			int m = processedData.getTotFeature();

			vector<long double> prob_feat(m, 0.0);
			for (int f = 0; f < m; f++)
				prob_feat[f] = (long double)__feat_sat[f] / (long double)__number_of_tries;
		

			// First = largest feature in processedData; tie-break by smallest selection probability.
			int first = 0;
			for (int f = 1; f < m; f++){
				int sz_f = processedData.getFeatSize()[f];
				int sz_cur = processedData.getFeatSize()[first];

				if (sz_f > sz_cur) 
					first = f;
				else if (sz_f == sz_cur && prob_feat[f] < prob_feat[first]) 
					first = f;
			}

			// Remaining features sorted by ascending selection probability
			vector<int> rest;
			rest.reserve(m - 1);
			for (int f = 0; f < m; f++)
				if (f != first) 
					rest.push_back(f);

			sort(rest.begin(), rest.end(), [&](int a, int b) { return prob_feat[a] < prob_feat[b]; });

			vector<int> order;
			order.reserve(m);
			order.push_back(first);
			order.insert(order.end(), rest.begin(), rest.end());
			return order;
		}


		int getNextFeature(const vector<int>& activeFeat, const vector<int>& blocked) const {

			const int m = data.getTotFeature();
			vector<char> isActive(m, 0);
			for (int f : activeFeat) 
				isActive[f] = 1;

			int best = -1;
			long double bestP = 2.0L;

			for (int f = 0; f < m; f++){
				if (isActive[f] || blocked[f]) continue;

				long double p = (long double)__feat_sat[f] / (long double)__number_of_tries;
				if (best < 0 || p < bestP) {
					best = f;
					bestP = p;
				}
			}

			return best; // may be -1 if none left
		}


		/**
		 * @brief Creates value-merging maps based on hardness of satisfaction.
		 *
		 * This heuristic function generates `valueMaps` to be used with the
		 * `PanelData(original, valueMaps)` constructor. It merges "easy" values
		 * (those with satisfaction prob below a threshold) into a single new value index.
		 *
		 * @param addedFeat Features already in the DP (will be skipped).
		 * @param[in,out] valueMaps The 2D map to populate. Will be initialized if empty.
		 * @param currentData The `PanelData` used to get scores.
		 * @param threshold The hardness threshold (or count of values to merge)
		 * for determining "easy" vs. "hard" values.
		 */
		void createValueMaps(vector<int> addedFeat,
			vector<vector<int>> &valueMaps,
			const PanelData &currentData, 
			double threshold){

			const int m = data.getTotFeature();
			
			if (valueMaps.empty()){
				valueMaps.assign(currentData.getTotFeature(), {});
				for (int f = 0; f < currentData.getTotFeature(); f++){
					valueMaps[f].assign(currentData.getFeatSize()[f], {});
					iota(valueMaps[f].begin(), valueMaps[f].end(), 0);
				}
			}


			for (int f = 0; f < m; f++){
				if (find(addedFeat.begin(), addedFeat.end(), f) != addedFeat.end())
					continue;


				const int featSz = data.getFeatSize()[f];

				vector<long double> pVal(featSz, 0.0L);
				for (int v = 0; v < featSz; v++)
					pVal[v] = (long double)__val_sat[f][v] / (long double)__number_of_tries;

				// Partition values into easy/hard using your threshold rule.
				vector<char> isEasy(featSz, 0);
				int easyCnt = 0, hardCnt = 0;
				for (int v = 0; v < featSz; v++) {
					if (pVal[v] >= (long double)threshold) {
						isEasy[v] = 1;
						easyCnt++;
					} else {
						hardCnt++;
					}
				}


				if (easyCnt <= 1) continue;
				if (hardCnt == 0) continue;
				if (hardCnt == featSz) continue;

				int nextId = 0;
				for (int v = 0; v < featSz; v++) {
					if (!isEasy[v]) 
						valueMaps[f][v] = nextId++;
				}
				const int easyBucket = nextId;
				for (int v = 0; v < featSz; v++) {
					if (isEasy[v]) valueMaps[f][v] = easyBucket;
				}
			}

			if (__verbose_mode) {
				cerr << YELLOW << "--- Value Maps (compression) ---" << WHITE << "\n";
				for (int f = 0; f < m; f++) {
					cerr << "Feature " << (char)('a' + f) << ": [ ";
					for (int v = 0; v < (int)valueMaps[f].size(); v++) {
						cerr << v << "->" << valueMaps[f][v] << " ";
					}
					cerr << "]\n";
				}
				cerr << YELLOW << "--------------------------------\n\n" << WHITE;
			}

		}


		

		/**
		 * @brief Implements the automatic feature selection heuristic.
		 *
		 * Tries to find the largest set of features for which the DP can be
		 * solved within a given time limit.
		 *
		 * 1. Calculates feature hardness and optimal order.
		 * 2. Optionally preprocesses data (merges values) based on heuristics.
		 * 3. Progressively builds DP instances, starting with the 2 hardest features.
		 * 4. Adds the next hardest feature and tries to solve, protected by a `runWatchdog` timer.
		 * 5. If a DP instance times out or runs out of memory, it stops.
		 * 6. Returns the last successfully completed DP instance.
		 *
		 * @param timeOut The time limit in minutes for *each attempt* to add a new feature.
		 * @return A `shared_ptr` to the largest `CountSets` instance that was
		 * successfully computed within the time limits.
		 * @throw std::runtime_error If the initial 2-feature DP fails.
		 */
		shared_ptr<CountSets> autoFeatureSelection(int timeOut = __dp_timeout){

			processedData = data;
			vector<vector<int>> valueMaps;
			vector<int> activeFeat;

			// Obtain feature hardness estimates
			runUniformDiagnostics(1'000'000);
			if (__verbose_mode) 
				printSelectionProbTable("k-uniform Satisf Prob");

			if (__preprocess) {
				createValueMaps(activeFeat, valueMaps, data, __preprocess_theshold);
				processedData = PanelData(data, valueMaps);
			}


			// Build initial order
			vector<int> order = chooseInitialOrder(processedData);

			if (__verbose_mode){
				cerr << YELLOW << "Initial order: " << WHITE;
				for (int f : order) cerr << (char)('a' + f) << " ";
				cerr << endl;
			}


			shared_ptr<CountSets> lastSuccessfulDP = nullptr;

			const int m = processedData.getTotFeature();
			vector<int> blocked(m, 0);


			


			while (activeFeat.size() < __maxFeatures){

				if (lastSuccessfulDP != nullptr) {
					// Feature prediction and early drop
					const long diagPanels = 10;
					const long long diagMaxSamples = 100'000;
					const long double easyAvgTries = 10'000.0L;  // max tries per panel

					if (__test_mode)
						cerr << BLUE << "Computing current satisfaction probabilities." << WHITE << endl;

					auto panels = samplingAlgorithm<false, false>(lastSuccessfulDP, diagPanels, diagMaxSamples);

					// if (__test_mode)
					// 	cerr << BLUE << "Number of successful samples: " << WHITE << panels.size() << endl;

					if (__verbose_mode)
						printSelectionProbTable("Current Satisfaction Probabilities", &activeFeat);

					// Early stop, if we can sample, lets sample
					if ((long)panels.size() == diagPanels){
						long double avgTries = (long double)__number_of_tries / (long double)diagPanels;
						if (__verbose_mode) {
							cerr << YELLOW << "Diagnostics avg tries/panel: " << WHITE
								 << (double)avgTries << endl;
						}
						if (avgTries <= easyAvgTries){
							if (__verbose_mode)
								cerr << YELLOW << "Sampling is already easy; stopping DP growth." << WHITE << endl;
							break;
						}
					}
				}		


				// Get next feature
				int next = -1;

				if (activeFeat.empty()){
					for (int f : order){
						if (!blocked[f]){
							next = f;
							break;
						}
					}
				}
				else{
					next = getNextFeature(activeFeat, blocked);
				}

				if (next < 0)
					break;


				while (next >= 0 && (int)activeFeat.size() < __maxFeatures){

					vector<int> nxtFeatures = activeFeat;
					nxtFeatures.push_back(next);

					__DPFinished = false;
					__stopDPTimeout = false;

					// starts watchdog to stop the dp after timeout
					shared_ptr<CountSets> nextDP = nullptr;
					bool success = false;
					
					thread watchdog(runWatchdog, timeOut);
					try {

						nextDP = make_shared<CountSets>(processedData, weights, nxtFeatures, lastSuccessfulDP);

						success = true;

					} catch (const DpTimeoutException& e){
						if (__verbose_mode)
							cerr << YELLOW << "Timeout: " << WHITE << "Could not add feature " << (char)('a' + next) << " within " << timeOut << " minute(s)." << endl;
						
						success = false;
					} catch (const DpStateException& e){
						if (__verbose_mode)
							cerr << YELLOW << "Limits: " << WHITE << "Could not add feature " << (char)('a' + next) << " within " << timeOut << " minute(s)." << endl;
						
						success = false;
					} catch (const std::bad_alloc& e){
						if (__verbose_mode)
							cerr << RED << "Out of Memory: " << WHITE << "Failed to allocate memory for the next DP table while adding feature " << (char)('a' + next) << "." << endl;
						
						success = false;
					} catch (const std::exception& e){
						if (__verbose_mode)
							cerr << RED << "Exception: " << WHITE << e.what() << " while adding feature " << (char)('a' + next) << "." << endl;
						
						success = false;	
					} catch (...){
						if (__verbose_mode)
							cerr << RED << "Unknown exception: " << WHITE " while adding feature " << (char)('a' + next) << "." << endl;

						success = false;	
					}

					__DPFinished = true;
					watchdog.join();
				

					if (success){
						lastSuccessfulDP = nextDP;
						activeFeat = nxtFeatures;
						countSets[nextDP->getActiveFeat()] = nextDP;
					
						try{

							lastSuccessfulDP->computeWeightedComb();

							if (__verbose_mode) {
								cerr << YELLOW << "\tAdded feature " << WHITE << (char)('a' + next) << endl;
								cerr << YELLOW << "\tNumber of valid panels: " << WHITE
									 << lastSuccessfulDP->getAllSets() << endl << endl;
							}
							
						} catch (const std::exception& e){
							if (__verbose_mode)
								cerr << RED << "Exception: " << WHITE << e.what() << " cannot run intermidiate number of solutions computation." << endl;
						}

						break;	
					}
					else{
						blocked[next] = 1;
						if (__verbose_mode)
							cerr << YELLOW << "Could not add feature " << WHITE
								 << (char)('a' + next) << YELLOW << " (trying next)." << WHITE << endl;

						next = getNextFeature(activeFeat, blocked);
					}
				}		
			}

			__stopDPTimeout = false;

			autoSelectedCountSets = lastSuccessfulDP;
			if (!autoSelectedCountSets){
				cerr << RED << "Error:" << WHITE << " No DP built." << endl;
				exit(1);
			}

			countSets.clear();
			autoSelectedCountSets->clearMemory();
			autoSelectedCountSets->computeWeightedComb();

			if (__verbose_mode) {
				cerr << YELLOW << "\nReturning largest successful DP\n" << WHITE;
				cerr << YELLOW << "\tFeatures: " << WHITE;
				for (int f : autoSelectedCountSets->getActiveFeat())
					cerr << (char)('a' + f) << " ";
				cerr << endl;
				cerr << YELLOW << "\tNumber of valid panels: " << WHITE
					 << autoSelectedCountSets->getAllSets() << endl << endl;
			}

			return autoSelectedCountSets;
		}





		/**
		 * @brief Samples panels using a brute-force rejection sampling method.
		 *
		 * This function is intended for testing and validation only. It repeatedly
		 * shuffles all respondents, picks the first `panSize`, and checks if
		 * that panel is valid using `data.isValid()`.
		 *
		 * @param nPanels The target number of valid panels to find.
		 * @return A vector of frequencies, where `vec[i]` is the count for respondent `i`.
		 */
		vector<long long> bruteRespFrequency(long long nPanels){
			long long n = data.getRespondents().size();
			vector<long long> respFreq(n, 0);

			vector<long long> indices(n, 0);
			iota(indices.begin(), indices.end(), 0);

			std::mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
			vector<int> panel(panSize);
			
			for (int k = 0; k < nPanels; k++){
				do {
					shuffle(indices.begin(), indices.end(), rng);
					
					for (int i = 0; i < panSize; i++)
						panel[i] = indices[i];

				} while (!data.isValid(panel));
				for (auto x : panel)
					respFreq[x]++;
			}

			return respFreq;
		}

		/**
		 * @brief Prints a panel's respondent IDs to std::cout in sorted order.
		 *
		 * @param panel The panel to be printed, represented as a
		 * std::vector of respondent IDs.
		 */

		void printPanel(vector<int> &panel){
			__nPanelsPrinted++;
			sort(panel.begin(), panel.end());

			ostream &out = __panelot_output ? cerr : cout;
			if (__panelot_output) out << "\nPanel: ";

			for (int i = 0; i < panel.size(); i++){
				if (i + 1 < panSize)
					out << panel[i] << ",";
				else
					out << panel[i] << endl;
			}
		}

		/**
		 * @brief Prints a formatted summary of the Phase 2 Rejection Funnel.
		 * 
		 * Hides DP-enforced features into a single "Base DP" column, and then 
		 * sequentially displays the survival probability (in scientific notation) 
		 * and estimated valid panels as each remaining feature is applied.
		 */
		void printSamplingInfo(const shared_ptr<CountSets> &dp, bool final, bool force){
			if (__number_of_tries <= 0) // Should not happen
				return;

			if (!final && !force){
				auto now = chrono::steady_clock::now();
				if (now < nextSamplingInfoPrint)
					return;

				nextSamplingInfoPrint =
					now + chrono::seconds(__sampling_update_seconds);
			}

			double survivalRate = ((double)__nPanelsPrinted / __number_of_tries) * 100.0;
					
			ios_base::fmtflags f(cerr.flags());

			if (final)
				cerr << BLUE << "\n--- Final Rejaction Sampling Info ---" << WHITE << endl;
			else	
				cerr << BLUE << "\n--- Rejaction Sampling Info ---" << WHITE << endl;
			
			cerr << BLUE << "current # of panels: " << WHITE << __nPanelsPrinted 
				 << " (" << scientific << setprecision(2) << survivalRate << "%)" << endl;
			cerr.flags(f);

			cerr << BLUE << "current # of samples: " << WHITE << __number_of_tries << endl;

			// 1. Identify DP vs Non-DP features
			vector<int> activeFeat = dp->getActiveFeat();
			vector<bool> isDP(data.getTotFeature(), false);
			
			sort(activeFeat.begin(), activeFeat.end());
			string dpFeatStr = "";
			for (size_t i = 0; i < activeFeat.size(); i++) {
				isDP[activeFeat[i]] = true;
				dpFeatStr += string(1, 'a' + activeFeat[i]);
				if (i < activeFeat.size() - 1) dpFeatStr += ", ";
			}
			
			// Gather the remaining (Phase 2) features in checking order
			vector<int> remFeat;
			for (int i = 0; i < data.getTotFeature(); i++) {
				if (!isDP[i]) {
					remFeat.push_back(i);
				}
			}

			// Format column widths dynamically based on the DP string length
			int dpColW = max(14, (int)dpFeatStr.length() + 2);
			const int colW = 10;

			// --- Print Row 1: Headers ---
			cerr << setw(12) << right << "Feature" << " |";
			cerr << setw(dpColW) << right << dpFeatStr << " |";
			for (int f : remFeat) {
				cerr << setw(colW) << right << ("+" + string(1, 'a' + f)) << " |";
			}
			cerr << endl;

			// --- Print Row 2: Satisfaction Probability ---
			cerr << setw(12) << right << "Sat Prob" << " |";
			
			// Print base probability in scientific notation (1.00e+00)
			stringstream ssBaseProb;
			ssBaseProb << scientific << setprecision(2) << 1.0;
			cerr << setw(dpColW) << right << ssBaseProb.str() << " |";
			
			long long survivors = __number_of_tries;
			vector<double> probs;
			for (int f : remFeat) {
				// Subtract the panels that died EXACTLY at this feature
				survivors -= __panels_pruned[f];
				if (survivors < 0) survivors = 0; // Safety clamp
				
				double prob = 0.0;
				if (__number_of_tries > 0) {
					prob = (double)survivors / __number_of_tries; // Raw probability [0.0, 1.0]
				}
				probs.push_back(prob);
				
				stringstream ss;
				if (prob > 0) {
					ss << scientific << setprecision(2) << prob;
				} else {
					ss << "0.00e+00";
				}
				cerr << setw(colW) << right << ss.str() << " |";
			}
			cerr << endl;

			// --- Print Row 3: Estimated Panels ---
			cerr << setw(12) << right << "Est Panels" << " |";
			double base_panels = (double)dp->getAllSets();
			
			stringstream ssBase;
			ssBase << scientific << setprecision(1) << base_panels;
			cerr << setw(dpColW) << right << ssBase.str() << " |";

			for (double p : probs) {
				// We no longer divide by 100.0 because `p` is already a fraction
				double est = p * base_panels; 
				stringstream ss;
				if (est > 0) {
					ss << scientific << setprecision(1) << est;
				} else {
					ss << "0";
				}
				cerr << setw(colW) << right << ss.str() << " |";
			}
			cerr << endl;

			// --- Print Row 4: Attempts surviving each step ---
			cerr << setw(12) << right << "Attempts" << " |";

			// Base DP column: all attempts survive
			stringstream ssBaseAttempts;
			ssBaseAttempts << __number_of_tries;
			cerr << setw(dpColW) << right << ssBaseAttempts.str() << " |";

			long long attSurvivors = __number_of_tries;
			for (int f : remFeat) {
				attSurvivors -= __panels_pruned[f];
				if (attSurvivors < 0) attSurvivors = 0;
				
				stringstream ss;
				ss << attSurvivors;
				cerr << setw(colW) << right << ss.str() << " |";
			}
			cerr << endl;
		}

		/** @brief Gets the total number of respondents loaded. */
		inline int getNumResp(){return data.getRespondents().size(); }

		/** @brief Gets the target panel size. */
		inline int getPanSize(){return panSize;}
};

