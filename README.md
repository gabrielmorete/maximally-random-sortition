# Uniform Sortition

A tool for selecting panels uniformly from a pool of respondents.
Written in C++20.
---

### Compilation

To compile:

```bash
$ make main
```

### Basic Usage

To run the program, you must provide the data path and the target panel size.

```bash
$ ./main -path <data_dir> -size <n> [options]
```

Example:

```bash
$ ./main -size 30 -path data_panelot/pool_30/ -sample 100 -verbose -threads 5
```

-----

### Required Arguments

    `-path <path>`
            Path to the data directory. This directory must contain two files:
                    `categories.csv`: Defines features, values, and quota bounds.
                    `respondents.csv`: Contains the feature values for every candidate in the pool.
    `-size <n>`
            The target number of respondents in the panel.

-----

### Operating Modes

The program operates in different modes.

#### 1\. Counting

Calculates the exact number of valid panels that satisfy the quotas for specific features.

    `-count <f1 f2 ...>`
            Specify the features (by name, e.g., `a`, `b`) to include in the constraint satisfaction problem.
            If combined with `-sample`, the sampling is performed specifically on the solution space satisfying these features.

#### 2\. Sampling

Generates valid panels using a rejection sampling algorithm built on top of the DP graph.

    `-sample <n>`
            Generates `<n>` valid panels.
            If no number is provided, it defaults to 1.
            Output is printed to `stdout` in CSV format.


#### 3\. Approaching Target Marginals

Finds a set of respondent weights such that the weighted sampling distribution matches a specific target distribution (e.g., ensuring every respondent has the same probability of being selected as in the given distribution).

    `-tDist <file>`
            Path to a CSV file containing the target marginal probabilities for each respondent.

-----

### Configuration & Tuning

    `-threads <n>`
            Number of threads to use for parallel sampling and graph construction (OpenMP). Default is 1.
    `-preprocess <threshold>`
            Enables data preprocessing.

### Testing & Benchmarking

    `-verbose`
            Prints extra information to `stderr`.
    `-test`
            Prints detailed execution logs to `stderr`.

-----

### Input File Formats

#### 1\. categories.csv

Defines the quotas.

    Header: `FeatureName,ValueName,MinQuota,MaxQuota`
    Example:
        ```csv
        FeatureName,ValueName,MinQuota,MaxQuota
        gender,Male,14,16
        gender,Female,14,16
        age,Young,5,10
        ```

#### 2\. respondents.csv

Defines the pool of candidates.

    Header: `ID,FeatureName1,FeatureName2,...`
    Example:
        ```csv
        ID,gender,age
        0,Male,Young
        1,Female,Young
        ```

#### 3\. Target Distribution (for `-tDist`)

    Header: `ID,Probability` (implied)
    Format:
        ```csv
        Probability,Member1,Member2,Member3
        0.1,5,10,15
        0.05,2,5,8
        ...
        ...
        ```

<!-- end list -->

```
```
