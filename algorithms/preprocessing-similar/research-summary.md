# Research Summary: Foundations of Data Mining & Similarity Selection
> **Theoretical grounding for the `oxidized-mining` implementation.**

## Executive Summary
This research analyzes the critical impact of data preprocessing on the structural integrity of a dataset. Using the Titanic dataset as a case study, the work evaluates how systematic handling of missingness, outliers, and feature scaling establishes a reliable foundation for subsequent similarity and distance analysis.

## Theoretical Frameworks

### 1. CRISP-DM Methodology
This exploration follows the **Cross-Industry Standard Process for Data Mining (CRISP-DM)**. By strictly adhering to the Data Understanding and Data Preparation phases, this project ensures that the "messy" reality of raw data—characterized by substantial missingness in the `Age` and `Cabin` fields—is addressed before any modeling occurs.

### 2. Probably Approximately Correct (PAC) Learning
The implementation is informed by the **PAC Learning framework**, which provides a mathematical lens for evaluating hypothesis generalization. In a dataset of 891 observations, minimizing the error introduced during preprocessing is essential to ensuring that the resulting similarity metrics remain "probably" correct and representative of the underlying population.

---

## Statistical Analysis & Data Fidelity

### Distribution and Skewness
A key finding in the research was the high right-skewness of the `Fare` attribute. Without intervention, such skewed distributions can dominate distance-based algorithms. 
* **Action:** Applied Z-score standardization to transform numerical features into mean-zero, unit-variance scaled representations.
* **Result:** Enabled smoother similarity gradients across the feature space.

### Dimensionality Reduction
To address high dimensionality and noise, **Principal Component Analysis (PCA)** was utilized to reduce the feature set to its primary components while retaining maximum variance. This step is critical for visualizing passenger patterns in a lower-dimensional space.

---

## Evaluation of Similarity Metrics
The choice of a distance metric is a core determinant of analytical validity. This research contrasts three primary approaches:

* **Euclidean Distance:** Selected for its reliability in standardized numeric spaces, particularly for capturing magnitude-based relationships between passengers.
* **Cosine Similarity:** Utilized for its ability to reflect directional patterns rather than absolute magnitudes, making it well-suited for high-dimensional or directional feature relationships.
* **Jaccard Similarity:** Reserved for engineered binary features (e.g., the `LargeFamily` indicator). Unlike Euclidean distance, Jaccard is optimized for categorical overlaps and ignores "shared absences," which can otherwise distort similarity scores in sparse data.

---

## Engineering Implications
The "Oxidization" of these algorithms into Rust demonstrates that preprocessing is not merely a preparatory step but a deterministic phase of the engineering lifecycle. By implementing these metrics manually, we ensure that the mathematical logic is decoupled from high-level library assumptions, allowing for greater precision in neighbor selection and model interpretability.

***
