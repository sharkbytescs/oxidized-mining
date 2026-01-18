# Exploration: Data Preprocessing & Similarity Metrics
> **A Case Study: Bridging Exploratory Data Analysis (Python) and High-Performance Engineering (Rust)**

## Overview
This exploration focuses on the essential first step: preparing data for discovery. Using the Titanic dataset, I document the process of cleaning and transforming raw variables into meaningful features—a phase where data quality dictates the geometry of the feature space.

From an engineering perspective, this is a study in functional data integrity. I analyze how foundational operations—such as iterative imputation and non-linear transformations—directly redefine the Euclidean and Jaccard distances between observations. 
By 'oxidizing' these prototypes into Rust, I demonstrate that the mathematical accuracy of an algorithm is fundamentally coupled with the precision of the preprocessing pipeline.

---

## 🛠 The Implementation Workflow

### Phase 1: Exploratory Data Analysis (Python)
Using the industry-standard stack, I performed a deep dive into data quality and feature extraction.
* **Libraries:** `Pandas`, `Seaborn`, `Scikit-Learn`, `SciPy`.
* **Key Tasks:**
    * **Iterative Imputation:** Handling missing `Age` values using title-based grouping.
    * **Statistical Profiling:** Measuring Skewness and Kurtosis to determine if Log Transformations were required for `Fare` data.
    * **Dimensionality Reduction:** Applying **Principal Component Analysis (PCA)** to visualize variance across features.

### Phase 2: Algorithmic "Oxidization" (Rust)
To showcase engineering rigor, I rewrote core similarity logic in Rust. This moves away from "black-box" libraries to demonstrate a first-principles understanding of linear algebra.
* **Focus:** Zero-cost abstractions and memory safety.
* **Algorithms Implemented:**
    * **Jaccard Similarity:** Optimized for binary categorical features (e.g., the `LargeFamily` feature).
    * **Euclidean Distance:** Implemented with Rust iterators for high-performance vector calculations.

---

## 🔬 Theoretical Anchors
This exploration is grounded in several academic frameworks, detailed in the accompanying research summary:
1. **CRISP-DM:** Standardizing the lifecycle of data mining.
2. **PAC Learning:** Understanding the bounds of "Probably Approximately Correct" learning in the context of sample size.
3. **Metric Theory:** Comparing Euclidean vs. Jaccard distances for high-dimensional vs. sparse binary data.

[Image of a data preprocessing and machine learning pipeline diagram]

---

## 📂 Folder Structure
* `python/`: Contains the original `.ipynb` notebook showing the EDA process.
* `rust/`: A standalone Cargo project with manual implementations of similarity metrics.
* `research-summary.md`: A derivative summary of the academic paper associated with this work.

## 🚀 Usage
**Python:** Run `jupyter notebook` within the `/python` directory.  
**Rust:** Run `cargo run` within the `/rust` directory to see the similarity matrix output.

---
**Author:** Mark Babcock  
*PhD Student in Computer Science*
