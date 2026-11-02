# Edu-Predict: Decision Tree Classifier from Scratch

A custom, interpretable **Decision Tree classifier** for predicting student exam outcomes (**Pass** or **Fail**) using the *Students Performance in Exams* dataset. The tree-building and prediction algorithm is implemented **from scratch**, without `scikit-learn`'s `DecisionTreeClassifier`.

## Project objective

This project demonstrates how a decision tree selects splits using **probability**, **entropy**, and **information gain**, and how its predictions can be evaluated and explained. It also explores Gini impurity and the combinatorics of candidate splits.

> **Interpretation caveat:** The target is calculated directly from the three exam scores. When those scores are used as inputs, the model is learning to approximate a known grading rule, **not** predicting performance before the exams. The five-feature experiment without scores provides a more limited comparison.

## Dataset

Source: [Kaggle — Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

The dataset contains **1,000 student records** and eight original input features:

| Feature | Meaning | Type |
| --- | --- | --- |
| `gender` | Student gender | Categorical |
| `race/ethnicity` | Group classification | Categorical |
| `parental level of education` | Parent's education level | Categorical |
| `lunch` | Lunch category | Categorical |
| `test preparation course` | Test preparation status | Categorical |
| `math score` | Mathematics exam score | Numerical |
| `reading score` | Reading exam score | Numerical |
| `writing score` | Writing exam score | Numerical |

### Target variable

The new target `final_grade` is **1 (Pass)** if the average score is **strictly greater than 60**, otherwise **0 (Fail)**:

```python
df["final_grade"] = (
    (df["math score"] + df["reading score"] + df["writing score"]) / 3 > 60
).astype(int)
```

An average of exactly 60 is **Fail (0)**.

## Data preprocessing

1. Load `data/StudentsPerformance.csv` with Pandas.
2. Create the target `final_grade`.
3. Split into **80% training / 20% test**, with `random_state=42` (**800 / 200 records**).
4. Map each categorical feature to integer codes; derive category mappings from the training partition to avoid fitting preprocessing on the test partition.
5. Compare two feature configurations:
   - **5 features:** the five categorical background/preparation features.
   - **8 features:** the same five features plus math, reading, and writing scores.

**Categorical-variable limitation:** The current implementation converts categories to integer labels and searches numerical `<=` thresholds across them. This works computationally, but it imposes an artificial order on nominal categories (e.g., `race/ethnicity`). A dedicated categorical subset split would be a meaningful future improvement.

## Mathematical background

### Probability

For class \(c\) in a node with \(n\) samples, its empirical probability is:

\[
p(c) = \frac{\text{number of samples of class }c}{n}
\]

### Entropy

Entropy measures class uncertainty:

\[
H(S) = -\sum_c p(c)\log_2 p(c)
\]

Terms with \(p(c)=0\) contribute zero. Entropy is **0** for a pure node and **1** for an evenly split binary node.

### Gini impurity

Gini measures class mixture in another way:

\[
Gini(S) = 1 - \sum_c p(c)^2
\]

For two balanced classes, Gini is **0.5**. The project implements Gini for comparison, while **entropy-based information gain** is the actual split criterion.

### Information gain

For a split into child partitions \(S_v\):

\[
IG(S,A) = H(S) - \sum_v \frac{|S_v|}{|S|}H(S_v)
\]

The algorithm chooses the candidate split with the **largest positive information gain**.

## Combinatorics and computational complexity

At a node with **\(d\)** input features, the implementation considers up to **\(d\) features** (here, **8** in the full model). A feature with **\(u_j\)** distinct observed values supplies up to **\(u_j - 1\)** candidate thresholds: the midpoints between sorted, consecutive unique values.

Therefore, the maximum number of threshold candidates inspected at a node is:

\[
T = \sum_{j=1}^{d}\max(u_j-1,0)
\]

For example, 10 distinct values in one feature give **9** candidate thresholds.

A separate combinatorics observation: selecting any nonempty subset from \(d\) features gives \(2^d-1\) possible subsets (255 for 8 features). **This implementation does not enumerate these feature subsets**; it tests one feature and one threshold at a time.

With **\(n\)** rows at a node, checking a threshold requires scanning and splitting the rows, giving approximately **\(O(nT)\)** split-search work (plus costs such as sorting unique values). Since \(T\le d(n-1)\), a simple worst-case bound is approximately **\(O(dn^2)\) per node** for this unoptimized implementation. Total training cost depends on the number and sizes of nodes. `max_depth` and `min_samples_split` limit tree growth.

## Custom Decision Tree implementation

The model is defined in `model/decision_tree.py` and includes:

- A `Node` data class storing `prediction`, `feature_index`, `threshold`, `left`, and `right`.
- Exhaustive candidate threshold search and entropy-based information gain.
- Recursive tree construction.
- Stopping conditions: pure labels, maximum depth, minimum samples, or no positive information gain.
- `fit()`, `predict()`, and a text display of the decision rules.

The experiments use:

```python
DecisionTree(max_depth=5, min_samples_split=2)
```

The root of the eight-feature tree splits on **`reading score <= 61.5`**. All leaf predictions are **0 (Fail)** or **1 (Pass)**.

## Evaluation results

Test set: **200 students**.

| Method | Test accuracy | Correct predictions |
| --- | ---: | ---: |
| Custom Decision Tree — 5 features | 67.50% | 135 / 200 |
| Custom Decision Tree — 8 features | **97.50%** | **195 / 200** |
| Majority-class dummy baseline | 71.00% | 142 / 200 |

The majority-class baseline always predicts the most frequent class observed in the **training set**.

### Confusion matrix — eight-feature model

Rows represent actual classes; columns represent predicted classes.

| | Predicted Fail (0) | Predicted Pass (1) |
| --- | ---: | ---: |
| **Actual Fail (0)** | **56** (TN) | **2** (FP) |
| **Actual Pass (1)** | **3** (FN) | **139** (TP) |

- **True negatives:** 56
- **False positives:** 2
- **False negatives:** 3
- **True positives:** 139
- **Total errors:** 5

The eight-feature tree exceeds the project's suggested **75% accuracy** target, but this result must be interpreted in light of the target's dependence on exam-score inputs.

## Additional analyses and saved results

The notebook generates plots and exports summary files to `results/`. The figures below are relative links, so they also work when the repository is viewed on another computer or in Gitea.

### 1. Decision tree visualization — `tree_visualization.png`

![Custom Decision Tree](results/tree_visualization.png)

The diagram shows the learned split conditions and predicted classes. Blue nodes contain questions; green leaves predict **Pass (1)** and pink leaves predict **Fail (0)**. Branch labels **Yes** and **No** indicate whether the condition holds. The root condition of the eight-feature tree is **`reading score <= 61.5`**.

### 2. Confusion matrix — `confusion_matrix.png`

![Confusion Matrix](results/confusion_matrix.png)

Displays the **56 true negatives**, **2 false positives**, **3 false negatives**, and **139 true positives** from the eight-feature model (depth 5), for **195 correct predictions out of 200**.

### 3. Class distribution — `class_distribution.png`

![Class Distribution](results/class_distribution.png)

Shows the target distribution across all 1,000 records: **293 Fail (29.3%)** and **707 Pass (70.7%)**. Because Pass is more common, accuracy should be compared against a majority-class baseline.

### 4. Tree depth comparison — `depth_comparison.png`

![Accuracy by Maximum Tree Depth](results/depth_comparison.png)

Compares **training and test accuracy** for maximum tree depths **1–10**. In the reported experiment, the highest observed test accuracy was **98.0% at depth 4** (196/200 correct), while the primary depth-5 tree achieved **97.5%** (195/200 correct). Deeper trees reached 100% training accuracy without increasing test accuracy. This is an exploratory comparison on the test set, **not a validated hyperparameter selection**; in a stricter workflow, choose depth using a separate validation set or cross-validation on training data and reserve the test set for the final report.

### 5. Two-dimensional decision boundary — `decision_boundary_2d.png`

![2D Decision Boundary](results/decision_boundary_2d.png)

An **independently trained** depth-4 tree uses only `math score` and `reading score` to show predicted Pass/Fail regions on a two-dimensional grid. Its test accuracy was **95.5%**, and should not be confused with the eight-feature model's 97.5% accuracy. The region plot shows the two-feature model's predictions, while the overlaid points represent actual test labels.

### 6. Machine-readable metrics — `metrics.json`

Stores accuracy values, the selected best depth from the exploratory depth comparison, the four confusion-matrix counts, and the Pass/Fail class distribution. Values are stored as fractions (for example, `0.975` means **97.5%**). Useful for programmatic inspection without rerunning the analysis.

### 7. Text decision rules — `tree_structure.txt`

A human-readable, indented export of the depth-5 tree's `if/else` conditions and leaf predictions. It provides a text alternative to the graphical visualization.

## Project structure

```text
edu-predict/
├── data/
│   └── StudentsPerformance.csv
├── model/
│   └── decision_tree.py
├── notebook/
│   └── analysis.ipynb
├── utils/
│   └── metrics.py
├── results/
│   ├── tree_visualization.png
│   ├── confusion_matrix.png
│   ├── class_distribution.png
│   ├── depth_comparison.png
│   ├── decision_boundary_2d.png
│   ├── metrics.json
│   └── tree_structure.txt
├── README.md
├── requirements.txt
└── .gitignore
```

## How to run the project

### Prerequisites

- **Python 3.11+** (a recent, supported version; the project was run with Python 3.14.7).
- **Git** to clone the repository, and **VS Code** with the Python and Jupyter extensions (recommended).
- The dataset file `data/StudentsPerformance.csv` included in the repository.

### Step 1 — Get the project

Clone your own repository from Tomorrow School Gitea, or download and extract the project ZIP. In a terminal, navigate to the **`edu-predict` project folder**, where `requirements.txt` is located.

```bash
cd path/to/edu-predict
```

Replace `path/to/edu-predict` with your actual folder location; **do not use `/Users/Guest/...` on another computer**.

### Step 2 — Create and activate a virtual environment

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3 — Install requirements and Jupyter kernel

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install ipykernel
```

If `ipykernel` is already listed in `requirements.txt`, the last command is unnecessary. It is included separately here so the setup works with earlier copies of the repository as well.

### Step 4 — Open and run the notebook

1. Open the **whole `edu-predict` folder** in VS Code.
2. Open `notebook/analysis.ipynb`.
3. Click **Select Kernel** and choose the Python interpreter inside `.venv`.
4. **Important for the current notebook:** it loads data with `../data/StudentsPerformance.csv`. The notebook therefore needs its **current working directory to be `edu-predict/notebook`**. If it instead starts in the repository root, that relative path will fail. You can check with `Path.cwd()` in a notebook cell. If needed, change the working directory to the notebook folder *before* running all cells, or update the load path to use the project root.
5. Choose **Restart Kernel → Run All** and run the cells **in order**, including the added model-training, evaluation, and plot-export cells.
6. Verify the printed metrics and open the generated images in `results/`.

> **Portability note:** The export cells create output paths using `Path.cwd()` and, when the working directory is called `notebook`, its parent directory. They do **not** hard-code the Mac username or absolute path. Run from the project root or the `notebook` folder for outputs to go into the expected `results/` directory. The input CSV path described above is more restrictive and requires special attention. 

### Step 5 — Confirm expected results

When using the same dataset, preprocessing, random split, and model parameters, the reference run produced:

```text
Decision Tree (5 features): 67.50%
Decision Tree (8 features): 97.50%
Dummy Baseline: 71.00%
2D Decision Tree Accuracy: 95.50%
Best test accuracy in depth comparison: 98.00% at depth 4
```

The actual run will also display the confusion matrix, class counts, and diagrams. If you downloaded a repository copy that already includes `results/`, you can view the exported artifacts without rerunning the notebook; running it will regenerate or overwrite them.

### Troubleshooting

- **`ModuleNotFoundError`**: verify that `.venv` is selected and dependencies are installed; check the project module import path for `model.decision_tree` and `utils.metrics`.
- **`FileNotFoundError` for `../data/StudentsPerformance.csv`**: check the notebook working directory (Step 4).
- **`NameError` for `X_train_full`, `DecisionTree`, etc.**: restart the kernel and run notebook cells from top to bottom.
- **No plots in `results/`**: ensure all export cells have executed and the current directory is the project root or `notebook/`.

## Conclusions

- The classifier was implemented from scratch, using entropy-based information gain rather than a ready-made decision-tree estimator.
- Adding exam scores increased test accuracy from **67.50%** to **97.50%**, exceeding the **71.00%** dummy baseline.
- The tree can be explained through explicit decision thresholds and leaf predictions.
- The experiment demonstrates both the value of informative features and the risk of treating a constructed target as an independent future outcome.

## Potential improvements

- Handle nominal categories via explicit equality or subset-based categorical splits.
- Compare different maximum tree depths on a validation split.
- Extend the two-feature decision-boundary experiment with additional projections.
- Add reusable tests for entropy, information gain, and tree prediction.
- Evaluate genuine early-risk prediction using only information available **before** final exam scores are known.
