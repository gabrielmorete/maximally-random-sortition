/**
 * @file state.h
 * @brief Defines the DPState struct used to represent states in the dynamic programming algorithm.
 */

#pragma once

/**
 * @struct DPState
 * @brief Represents a state within the dynamic programming.
 *
 * A DP state typically captures the current progress through the respondent tuples
 * and the aggregated counts (represented by FeatureMask IDs) for the features being tracked.
 *
 * @note Hard cap: Maximum number of features tracked in a single DP instance
 * is `__maxFeatures` (currently 8) to optimize memory usage of the state struct.
 */
struct DPState{

	/** @brief Id of the feature-value vector associated with this state*/
	uint16_t tupleId;

	/** @brief Current count specifically for the primary feature's (feature 0) active value. */
	uint16_t v0;
	
	/** @brief Number of active features in this state*/
	uint8_t numFeat; // Number of active features in this state.
	
	/**
	 * @brief Array storing the unique IDs (hashes) of the FeatureMasks for features 1 to numFeat-1.
	 * Feature 0's state (`v0_`) is stored separately and not hashed here.
	 * The size is __maxFeatures-1 because feature 0 is handled separately.
	 */
	uint32_t maskIds[__maxFeatures - 1]; // Value 0 does not have a hash
	
	// Constructor
	DPState() : tupleId(-1), v0(0), numFeat(0){
		memset(maskIds, 0, sizeof(maskIds));
	}

	bool operator==(const DPState& other) const {
		if (tupleId != other.tupleId || v0 != other.v0 || numFeat != other.numFeat)
			return false;
		

		if (numFeat == 0)
			return true;

		// Since we store the masks in a C-style array, we can use memcomp
		return memcmp(maskIds, other.maskIds, sizeof(uint32_t) * numFeat) == 0;
	}

	// To use with std::map
	bool operator<(const DPState& other) const {
		if (tupleId != other.tupleId)
			return tupleId < other.tupleId;
		if (v0 != other.v0)
			return v0 < other.v0;

		// If the above are equal, compare the mask arrays
		return std::lexicographical_compare(
			maskIds, maskIds + numFeat - 1,
			other.maskIds, other.maskIds + other.numFeat - 1
		);
	}
};

// Boost hash specialization
namespace boost {
	template<>
	struct hash<DPState> {
		size_t operator()(const DPState& s) const {
			size_t seed = 0;

			hash_combine(seed, s.tupleId);
			hash_combine(seed, s.v0);
			hash_combine(seed, s.numFeat);
			hash_combine(seed, boost::hash_range(s.maskIds, s.maskIds + s.numFeat));

			return seed;
		}
	};
}
