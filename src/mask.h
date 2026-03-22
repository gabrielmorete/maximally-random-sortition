/**
 * @file mask.h
 * @brief Defines the FeatureMask class for storing feature counts.
 *
 * This file contains the definition of the FeatureMask class, 
 * which stores counts for different values of a single feature 
 * within a panel.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <cassert>
#include <boost/functional/hash.hpp>

/**
 * @brief Stores and manages the counts for each possible value of a single feature.
 *
 * This class acts like a specialized bitmask or frequency array, optimized for memory
 * usage and hashing. It tracks the number of respondents (panel members) a feature.
 *
 * Hard Limits:
 * 	- Maximum number of distinct values per feature: `__maxFeatureValues` (currently 64).
 * 	- Maximum count per value (and thus max panel size): 65'535 (due to `uint16_t`).
 */


#pragma once

#include <cstdint>
#include <cstring>
#include <cassert>
#include <boost/functional/hash.hpp>

class FeatureMask{
private:
	/* Storage for counts. 
	   Using uint16_t allows counts up to 65'535 per value.
	*/
	uint16_t storage[__maxFeatureValues];
	
	/* Index of the feature in the feature vector that corresponds to this mask. */	
	uint8_t featureIdx;

	/* Number of values in this mask. */	
	uint8_t numValues; 
	
	/* Total number of respondents. */
	uint16_t currentSize; 
	
public:

	// Default Constructor
	FeatureMask() : featureIdx(0), numValues(0), currentSize(0){
		std::memset(storage, 0, sizeof(storage));
	}
	
	// Parameterized Constructor
	FeatureMask(int featureIdx, int values) 
		: featureIdx(static_cast<uint8_t>(featureIdx)), 
		  numValues(static_cast<uint8_t>(values)), 
		  currentSize(0) {
		assert(values <= __maxFeatureValues && "Number of values exceeds maximum.");
		std::memset(storage, 0, sizeof(storage));
	}

	/**
	 * @brief Increments the count for a specific value index.
	 */
	void incrementValue(int valueIdx, int amount = 1){
		assert(valueIdx < numValues && "Index out of bounds.");
		storage[valueIdx] += amount;
		currentSize += amount;
	}
	
	/**
	 * @brief Gets the current count for a specific value index.
	 * @return count as uint16_t
	 */
	uint16_t getCount(int valueIdx) const{
		assert(valueIdx < numValues && "Index out of bounds.");
		return storage[valueIdx];
	}
	
	// Size of the mask
	inline uint16_t getSz() const { return currentSize; }
	
	/**
	 * @brief Checks if the current counts satisfy given minimum and maximum quotas.
	 */
	template<typename T>
	bool isValid(const T& minV, const T& maxV) const{
		for (int i = 0; i < numValues; i++){
			if (storage[i] < minV[i] || storage[i] > maxV[i]) 
				return false;
		}
		return true;
	}

	/**
	 * @brief Calculates the total deficiency relative to minimum quotas.
	 */
	template<typename T>
	long long getDeficiency(const T& minV) const {
		long long def = 0;
		for (int i = 0; i < numValues; i++){
			if (storage[i] < minV[i])
				def += (minV[i] - storage[i]);
		}
		return def;
	}

	/**
	 * @brief Calculates the total slack relative to maximum quotas.
	 */
	template<typename T>
	long long getSlack(const T& maxV) const {
		long long slack = 0;
		for (int i = 0; i < numValues; i++)
			slack += (maxV[i] - storage[i]);
		
		return slack;
	}

	bool operator==(const FeatureMask& other) const {
		if (featureIdx != other.featureIdx || 
			numValues != other.numValues || 
			currentSize != other.currentSize) {
			return false;
		}
		
		// Compare the used portion of the storage.
		// Multiply by sizeof(uint16_t) because memcmp works in bytes.
		return std::memcmp(storage, other.storage, numValues * sizeof(uint16_t)) == 0;
	}
	
	// Friend declaration for boost hash
	friend struct boost::hash<FeatureMask>;

	const int getFeatureId() const {return featureIdx;}
};

// Boost hash specialization
namespace boost {
	template<>
	struct hash<FeatureMask>{
		size_t operator()(const FeatureMask& m) const {
			size_t seed = 0;
			
			boost::hash_combine(seed, m.featureIdx);
			boost::hash_combine(seed, m.numValues);
			boost::hash_combine(seed, m.currentSize);
			
			// Hash the range of uint16_t values
			// storage is the start pointer, storage + numValues is the end pointer
			boost::hash_range(seed, m.storage, m.storage + m.numValues);

			return seed;
		}
	};
}