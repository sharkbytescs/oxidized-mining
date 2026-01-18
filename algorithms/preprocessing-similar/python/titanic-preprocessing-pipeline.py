import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.metrics import jaccard_score, pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, StandardScaler

titanic = pd.read_csv("titanic.csv")  # load the data

# Identify target column
# target_column = titanic.columns[1]   # "Survived"
y = titanic["Survived"]

# Print the number of rows and columns in titanic
print("Shape of titanic:", titanic.shape)

# ---------------------------------------------------------------
# Step 2: Examine Missingness BEFORE Applying Any Cleaning Steps
# ---------------------------------------------------------------
# It is critical to establish a baseline understanding of the raw
# dataset before performing any preprocessing. This step allows us
# to identify which variables contain missing values, how many
# values are missing, and the overall pattern of missingness.
#
# Capturing the "before" state is required for comparative analysis:
#   • It helps justify our choice of imputation strategies.
#   • It provides transparency in the preprocessing workflow.
#   • It allows us to document how each cleaning operation affects
#     data quality, which is essential for reproducibility.
#
# The code below computes the number of missing (NaN) values in
# each column of the dataset. This output will later be compared
# with the post-imputation results ("after" snapshots) to
# demonstrate the effectiveness of the cleaning steps.
# ---------------------------------------------------------------
missing_before = titanic.isnull().sum()
print(missing_before)

# -------------------------------------------------------------------------
# Visualizing Missingness with a Heatmap (Before Preprocessing)
# -------------------------------------------------------------------------
# While the previous table summarized the *count* of missing values in each
# column, a heatmap provides a far more intuitive, visual understanding of
# *where* missingness occurs across the dataset.
#
# Why this visualization is important:
#   • It reveals patterns of missing data (e.g., entire columns like "Cabin"
#     may be mostly empty, indicating structural missingness).
#   • It helps determine whether missingness is random or concentrated in
#     specific rows/variables.
#   • It supports evidence-based decisions when choosing imputation strategies.
#   • It fulfills the assignment requirement to provide *before-and-after*
#     snapshots of preprocessing results.
#
# Interpreting the plot:
#   • Yellow (or light-colored) cells indicate missing values (NaN).
#   • Dark cells indicate present/non-missing values.
#   • Columns with continuous yellow patterns may require removal or special
#     handling (e.g., "Cabin" in the Titanic dataset).
#
# This heatmap represents the "BEFORE" state of the dataset and will later be
# compared to an "AFTER" heatmap once missing values have been handled through
# imputation or deletion.
# -------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.heatmap(titanic.isnull(), cbar=False)
plt.title("Missing Values Before Preprocessing")
plt.show()

# ======================================================================
# Data Exploration: Identifying Numerical vs. Categorical Attributes
# Assignment Connection:
# This step supports the "Data Cleaning" and "Data Transformation" phases
# by identifying variable types prior to applying preprocessing techniques
# such as imputation, encoding, scaling, and outlier detection.
#
# Why this step matters:
# - Numerical features (int64/float64) require techniques like
#   standardization, normalization, and outlier detection.
# - Categorical features (object/category) require encoding (label or
#   one-hot encoding) before they can be used in machine learning models.
# - Many preprocessing steps must be applied *only* to specific types
#   of variables, so identifying them early ensures correct processing.
# ======================================================================

# Extract the list of numerical columns based on their data types.
numeric_cols = titanic.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Extract the list of categorical columns based on object or category types.
categorical_cols = titanic.select_dtypes(
    include=["object", "category"]
).columns.tolist()

# Display the lists of numerical and categorical columns for verification.
print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# ----------------------------------------------------------------------
# Additional Summary:
# Counting the number of numerical and categorical variables provides
# a quick structural overview of the dataset. This is often included in
# professional data audit reports and ensures that preprocessing steps
# scale correctly across multiple features.
# ----------------------------------------------------------------------

num_categorical = len(categorical_cols)
num_numerical = len(numeric_cols)

print("Number of categorical variables:", num_categorical)
print("Number of numerical variables:", num_numerical)


print("Number of categorical columns:", num_categorical)
print("Number of numerical columns:", num_numerical)

# Display data types and count of unique types
print("Column Data Types:")
print(titanic.dtypes)

print("\nSummary of Data Types:")
print(titanic.dtypes.value_counts())

# ======================================================================
# Data Transformation: Converting Object Columns to Categorical Types
# Assignment Requirement:
# "Encode categorical attributes using one-hot encoding or label encoding."
#
# Purpose of This Step:
# Many columns in the Titanic dataset are initially stored as Python
# object types (strings). Converting them into pandas 'category' dtype:
#
# 1. Explicitly signals which variables are categorical.
#    - This improves clarity during preprocessing.
#    - Prevents numeric transformations (e.g., scaling) from being
#      incorrectly applied to non-numeric data.
#
# 2. Optimizes memory usage.
#    - 'category' uses far less memory than plain string/object types.
#
# 3. Simplifies downstream encoding.
#    - Once variables are converted to categorical, pandas provides
#      built-in tools (.cat.codes, CategoricalDtype) for easy label
#      encoding and analysis.
#
# 4. Ensures consistency.
#    - Having all categorical variables explicitly coded as 'category'
#      ensures your encoding step later only targets the correct columns,
#      which supports reproducibility and clean pipeline design.
# ======================================================================

# Create a working copy of the original dataset to preserve raw data.
titanic2 = titanic.copy()

# Identify all object-type columns and convert them to categorical dtype.
# This includes variables such as Name, Sex, Ticket, Cabin, Embarked, etc.
titanic2[titanic2.select_dtypes(include="object").columns] = titanic2.select_dtypes(
    include="object"
).astype("category")

# ----------------------------------------------------------------------
# Verification Step:
# We confirm that the previously object-type columns have successfully
# been converted to categorical types. This serves as a checkpoint before
# performing encoding in later steps.
# ----------------------------------------------------------------------
categorical_dtypes = titanic2.select_dtypes(include="category").dtypes
categorical_dtypes

# Count categorical and numerical variables in htrain
num_categorical2 = len(titanic2.select_dtypes(include=["object", "category"]).columns)
num_numerical2 = len(titanic2.select_dtypes(include=["int64", "float64"]).columns)

print("Number of categorical columns:", num_categorical2)
print("Number of numerical columns:", num_numerical2)

# Check missing values by column
missing_by_column = titanic2.isnull().sum().sort_values(ascending=False)
print("Missing Values by Column:\n", missing_by_column)

# Check missing values by row
missing_by_row = titanic2.isnull().sum(axis=1).sort_values(ascending=False)
print("\nMissing Values by Row:\n", missing_by_row)

# Prepare heatmap data: convert categorical to numeric codes, preserve NaNs
heatmap_df = titanic2.copy()
for col in heatmap_df.select_dtypes(include=["object", "category"]).columns:
    heatmap_df[col] = heatmap_df[col].astype("category").cat.codes
    heatmap_df[col] = heatmap_df[col].replace(-1, np.nan)  # -1 = missing in cat.codes

# Create and show the heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_df.isnull(), cbar=False, cmap="viridis", yticklabels=False)
plt.title("Missing Values Heatmap (Categorical + Numeric)")
plt.xlabel("Features")
plt.ylabel("Rows")
plt.tight_layout()
plt.show()

# Missing Value Treatment
# Assignment Step: "Apply at least two missing data handling strategies."

# Missing Data Handling Strategy 1: Missingness Indicator [Feature Engineering for Missingness]
# Create a binary indicator showing which rows originally had missing Age values.
# This preserves information about missingness, which can sometimes improve model performance.
titanic2["Age_missing"] = titanic2["Age"].isnull().astype(int)

# Missing Data Handling Strategy 2: Median Imputation [for numeric Age]
# Impute missing Age values using the median of the Age distribution.
# Median imputation is appropriate for skewed numeric distributions such as Age.
titanic2["Age"].fillna(titanic2["Age"].median(), inplace=True)

# Missing Data Handling Strategy 3: Mode Imputation [for categorical Embarked]
# Impute missing Embarked values using the most frequent category (mode).
# Mode imputation is a standard approach for categorical variables.
titanic2["Embarked"].fillna(titanic2["Embarked"].mode()[0], inplace=True)


# Verify missing values are resolved
print("\nMissing Values After Imputation:\n", titanic2.isnull().sum())

# -------------------------------------------------------------------------
# Visualizing Remaining Missingness After Applying Imputation Strategies
# -------------------------------------------------------------------------
# At this stage, several missing-data handling strategies have already been
# applied to the dataset, including:
#   • Creating an indicator variable (“Age_missing”) to preserve information
#     about rows that originally contained missing Age values.
#   • Median imputation for the Age attribute to address numerical missingness.
#   • Mode imputation for the Embarked attribute to handle categorical gaps.
#
# To verify the effectiveness of these preprocessing steps, we generate a
# heatmap showing any remaining missing values across all columns. This
# visualization functions as a post-imputation audit, ensuring that the
# cleaning strategies were successful and that no unexpected gaps remain.
#
# Why this visualization matters:
#   • Confirms that missingness has been reduced or eliminated as intended.
#   • Highlights whether additional cleaning steps are still needed.
#   • Provides the required *after-preprocessing* snapshot for comparison
#     against the earlier “before” heatmap.
#
# Interpreting the plot:
#   • Light-colored cells indicate any remaining missing values.
#   • Dark cells indicate complete, non-missing observations.
#
# A clean heatmap (i.e., no light-colored areas) demonstrates that the
# imputation process was correctly applied and that the dataset is now
# ready for subsequent preprocessing steps such as encoding, scaling,
# feature engineering, and dimensionality reduction.
# -------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.heatmap(titanic2.isnull(), cbar=False)
plt.title("Missing Values Before Preprocessing")
plt.show()

# ======================================================================
# Data Cleaning: Outlier Detection Using the Interquartile Range (IQR)
# Assignment Requirement:
# "Detect and handle outliers (use techniques like Z-score, IQR, or DBSCAN)."
#
# Purpose of this step:
# Outliers can distort statistical summaries, impact distance-based methods,
# and reduce the performance of many machine learning models. The Titanic
# dataset contains features (such as Fare) with strong right-skew due to
# extremely high first-class ticket prices. Identifying these outliers is
# essential for making informed decisions about data cleaning.
#
# Why use the IQR method?
# The IQR method is a robust, non-parametric technique that:
#   - does not assume normal distribution,
#   - is resistant to extreme values,
#   - works well for skewed data (like Fare),
#   - and is a widely accepted standard for introductory and advanced EDA.
#
# Definition:
#   IQR = Q3 – Q1
#   Outlier thresholds:
#       Lower Bound  = Q1 – 1.5 * IQR
#       Upper Bound  = Q3 + 1.5 * IQR
#
# Any value outside these bounds is flagged as a statistical outlier.
# ======================================================================

# Compute the first (Q1) and third (Q3) quartiles of the Fare distribution.
Q1 = titanic2["Fare"].quantile(0.25)
Q3 = titanic2["Fare"].quantile(0.75)

# Calculate the Interquartile Range (the spread of the middle 50% of data).
IQR = Q3 - Q1

# Determine the lower and upper bounds for identifying outliers.
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Identify rows where Fare falls outside the acceptable IQR range.
# These rows represent unusually low or unusually high ticket prices.
outliers = titanic2[(titanic2["Fare"] < lower) | (titanic2["Fare"] > upper)]

# Print the number of detected outliers and confirm the shape.
# This helps assess whether outliers are rare anomalies or represent
# structural differences such as first-class pricing.
print("Fare Outliers:", outliers.shape)

# ================================================================
# Outlier Handling: Winsorization (Capping Extreme Fare Values)
# Assignment Requirement:
# "Detect and handle outliers (use techniques like Z-score, IQR, or DBSCAN)."
#
# Purpose:
# After detecting outliers in the Fare variable using the IQR method,
# the next required step is to *handle* these outliers. Removing them
# would reduce the dataset size, while leaving them unchanged could
# distort distance-based metrics and PCA. Winsorization provides a
# balanced approach by capping extreme values without discarding data.
#
# Why Winsorization?
# - Fare is heavily right-skewed due to very expensive first-class tickets.
# - Capping extreme values reduces variance inflation.
# - Maintains the full dataset (important for Titanic’s already small size).
# - Improves robustness for modeling, clustering, and similarity metrics.
# ================================================================

# Before-handling summary for comparison
print("Summary of Fare BEFORE Outlier Handling:")
print(titanic2["Fare"].describe())

# Apply winsorization (capping)
titanic2["Fare"] = titanic2["Fare"].clip(lower=lower, upper=upper)

# After-handling summary for comparison
print("\nSummary of Fare AFTER Outlier Handling:")
print(titanic2["Fare"].describe())

# -------------------------------------------------------------------------
# Data Transformation: Encoding Categorical Features
# Assignment Requirement:
# "Encode categorical attributes using one-hot encoding or label encoding."
#
# Purpose:
# Many machine learning models require all inputs to be numeric. The Titanic
# dataset includes categorical attributes such as 'Sex' and 'Embarked', which
# must be converted into numerical form before model training.
#
# Label encoding assigns a unique integer to each category and works well for
# binary variables (e.g., Sex) or for models that can handle arbitrary
# categorical ordering. One-hot encoding is another valid approach, but here
# we apply label encoding for simplicity and interpretability.
#
# After this transformation, the dataset will be fully numeric and ready for
# further preprocessing steps (scaling, feature engineering, PCA, etc.).
# -------------------------------------------------------------------------

titanic_encoded = titanic2.copy()
le = LabelEncoder()

for col in ["Sex", "Embarked"]:
    titanic_encoded[col] = le.fit_transform(titanic_encoded[col])

titanic_encoded[["Sex", "Embarked"]].head()

# ============================================
# Data Transformation: Standardization Step
# Assignment Requirement:
# "Normalize or standardize numerical attributes
#  (choose min-max normalization or z-score standardization)."
# ============================================

# Initialize the StandardScaler, which performs Z-score standardization.
# Z-score standardization rescales each numeric feature so that it has
# a mean of 0 and a standard deviation of 1.
# This is useful for distance-based methods and PCA, and prevents features
# with large scales (e.g., Fare) from dominating those with smaller scales (e.g., Parch).
scaler = StandardScaler()

# Select the numerical columns to scale.
# PassengerId and Survived are excluded to avoid scaling identifiers or the label.
numeric_cols_to_scale = ["Age", "Fare", "Pclass", "SibSp", "Parch"]

# Create a copy of the dataset to store the scaled values.
titanic_scaled = titanic2.copy()

# Apply Z-score standardization to the selected numeric columns.
# fit_transform() computes the scaling parameters and applies them.
titanic_scaled[numeric_cols_to_scale] = scaler.fit_transform(
    titanic2[numeric_cols_to_scale]
)

# Display summary statistics *before* scaling.
# These values will show the original scales, ranges, and distributions.
print("Summary Statistics BEFORE Standardization:")
print(titanic2[numeric_cols_to_scale].describe())

# Display summary statistics *after* scaling.
# After standardization, each scaled column should have mean ≈ 0 and std ≈ 1.
print("\nSummary Statistics AFTER Standardization:")
print(titanic_scaled[numeric_cols_to_scale].describe())

# ===============================================================
# Before-and-After Comparison: Z-Score Standardization
# Assignment Requirement:
# "Provide before-and-after snapshots (tables or plots) showing
#  the changes resulting from data preprocessing."
#
# Purpose:
# After performing Z-score standardization using StandardScaler,
# the assignment requires a clear, side-by-side comparison of
# numeric features BEFORE and AFTER scaling. This demonstrates
# that the transformation step was correctly executed and that
# the standardized variables now share a common mean (≈0) and
# standard deviation (≈1), which is critical for distance-based
# methods, PCA, and many ML algorithms.
# ===============================================================

# Select a subset of numeric variables to display in the snapshot.
# These features are representative of different ranges and scales.
numeric_sample = ["Age", "Fare", "SibSp", "Parch"]

# Extract the first 5 rows from the original (cleaned) dataset
# to illustrate the raw, unscaled numeric values.
before_numeric = titanic2[numeric_sample].head()

# Extract the corresponding first 5 rows from the standardized dataset
# to illustrate the effect of Z-score normalization.
after_numeric = titanic_scaled[numeric_sample].head()

# Concatenate the two tables side-by-side with clear labels.
comparison_numeric = pd.concat(
    [before_numeric, after_numeric],
    axis=1,
    keys=["Before Scaling (Raw Values)", "After Scaling (Z-Scores)"],
)

print("\nNumeric Feature Scaling Comparison (Before vs After):")
print(comparison_numeric)

# ================================================================
# Data Transformation: Categorical Encoding
# Assignment Requirement:
# "Encode categorical attributes using one-hot encoding or label encoding."
#
# Explanation:
# Many machine learning algorithms require numerical inputs and cannot
# directly process categorical (string or category-type) variables.
# Therefore, categorical variables must be encoded into numerical form.
#
# In this step, we perform LABEL ENCODING, which assigns each category
# a unique integer value. This approach is appropriate for:
# - low- or medium-cardinality features (e.g., Sex, Embarked, Title)
# - tree-based models or distance metrics that can tolerate integers
# - preprocessing pipelines where simplicity is preferred
#
# NOTE:
# One-hot encoding could also be used, but label encoding is sufficient
# for this assignment and aligns with the requirement to encode categories.
# ================================================================

# Identify all categorical columns (dtype='category') in the dataset.
categorical_cols = titanic2.select_dtypes(include="category").columns

# Create a copy of the dataset to store the encoded results.
titanic_encoded = titanic2.copy()

# Convert each categorical column into integer codes.
# pandas.Categorical assigns a unique integer to each category label.
for col in categorical_cols:
    titanic_encoded[col] = titanic_encoded[col].cat.codes

# Display the first few rows of the encoded dataset
# to verify successful transformation.
print(titanic_encoded.head())

# ================================================================
# Data Reduction: Principal Component Analysis (PCA)
# Assignment Requirement:
# "Apply one data reduction technique such as sampling,
#  aggregation, Principal Component Analysis (PCA), or feature selection."
#
# Explanation:
# PCA is a dimensionality reduction technique that transforms the
# original numerical features into a smaller set of uncorrelated
# components (principal components). These components capture the
# maximum possible variance in the data using fewer dimensions.
#
# Why PCA is appropriate here:
# - The Titanic dataset contains several correlated numeric features
#   (e.g., SibSp and Parch are related, Age and Fare distributions vary).
# - PCA reduces dimensionality while preserving the most important
#   information (variance).
# - PCA is beneficial for visualization, noise reduction, and improving
#   computational efficiency in downstream tasks.
#
# NOTE:
# Standardization is required before PCA to ensure all features contribute
# equally. You have already standardized the numeric variables, which
# makes this step valid and meaningful.
# ================================================================

# Initialize PCA to reduce the dataset to 2 principal components.
# n_components=2 means we want the two directions of greatest variance.
pca = PCA(n_components=2)

# Select only numeric columns (integer and floating-point) for PCA.
# PCA cannot process categorical variables directly, so encoded numerical
# columns are used here.
numeric_data = titanic_encoded.select_dtypes(include=["int64", "float64"])

# Fit PCA to the numeric data and transform it into principal components.
# The result is a new matrix where each row is represented by only
# two values (PC1 and PC2) instead of the original high-dimensional space.
pca_components = pca.fit_transform(numeric_data)

# Display the shape of the PCA-transformed data.
# Expectation: (number_of_rows, 2) → confirming successful reduction.
print("PCA shape:", pca_components.shape)
