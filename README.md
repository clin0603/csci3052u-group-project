# Predicting Mortgage Application Outcomes

**Remedial Finance · CSCI3052U Machine Learning · Ontario Tech University**

Can four basic application features predict recorded mortgage approvals and denials across years? We compare four classifiers and a majority baseline using historical Nevada Home Mortgage Disclosure Act (HMDA) data. We train on 2007 applications, select model settings on 2008 applications, and evaluate the selected models on 2009 applications.

## Project workflow

The project began with exploratory analysis of Nevada mortgage applications, followed by baseline modelling with k-nearest neighbours and logistic regression. The final experiments add single and bagged decision trees, compare all models on the same held-out year, and examine uncertainty and classification errors.

**Start with the Milestone 5 final experiments notebook.** It is the final implementation for the whole project and runs the data preparation, model comparisons, and follow-up analyses.

## Quick start

1. Open the Milestone 5 final experiments notebook in Google Colab. The latest shared copy is `Milestone_5_Final_Experiments(4).ipynb`.
2. Run the cells from top to bottom. The data setup cell downloads the three yearly CSVs from this repository into `Random_Mortgage_Data/`. If the download is unavailable, place the CSVs in that folder manually before running the experiments.
3. View the tables and plots in the notebook. The notebook also writes results to `results/random/`.

The notebook uses Python 3, pandas, NumPy, scikit-learn, and Matplotlib.

## Dataset and experimental setup

We used a random sample of 40,000 Nevada HMDA records per year before filtering. HMDA action codes 1 (loan originated) and 2 (approved but not accepted) were grouped as **approved**. Code 3 was labelled **denied**. Other action codes were excluded. [CFPB historic HMDA data](https://www.consumerfinance.gov/data-research/hmda/historic-data/)

| Split | Input file in `Random_Mortgage_Data/` | Records after filtering |
| --- | --- | ---: |
| 2007 training | `Random_Mortgage_Data07.csv` | 26,486 |
| 2008 validation | `Random_Mortgage_Data08.csv` | 27,604 |
| 2009 test | `Random_Mortgage_Data09.csv` | 24,470 |

All trained models use loan amount, applicant income, loan type, and loan purpose. We fit imputation, numeric scaling, and one-hot encoding using the training data. We select hyperparameters by validation ROC-AUC and reserve the 2009 data for the main test comparison.

## Main results

| Model | Selected setting | 2009 ROC-AUC | Accuracy | Denial recall |
| --- | --- | ---: | ---: | ---: |
| Majority baseline | Predict approval for every application | 0.5000 | 74.29% | 0.00% |
| k-nearest neighbours | 101 neighbours | 0.6413 | 74.72% | **11.72%** |
| Logistic regression | L2 regularization, C = 0.1 | 0.6411 | 74.31% | 0.17% |
| Single decision tree | Maximum depth 5 | 0.6232 | 74.21% | 7.95% |
| Bagged decision trees | 100 trees, maximum depth 5 | **0.6424** | **74.84%** | 11.24% |

ROC-AUC measures how well a model ranks applications across thresholds. Denial recall is the share of recorded denials correctly identified under the default prediction rule. Bagged trees had the highest observed test ROC-AUC, and kNN caught the most denials. The paired bootstrap intervals for the ROC-AUC differences between bagged trees and each of kNN and logistic regression included zero.

### Follow-up analyses

- **Sensitivity and ablation.** The notebook varies kNN neighbourhood size, logistic regression regularization, and tree depth. It also compares the single tree with bagged trees at matched depths.
- **Repeatability and uncertainty.** Five bagged-tree refits use seeds 42–46. A paired bootstrap with 1,000 resamples estimates uncertainty in the 2009 results.
- **Error analysis.** The notebook compares classification errors, denial recall by loan purpose, and a logistic regression cutoff selected using 2008 data.

## Milestone 5 outputs

The repository includes the [saved Milestone 5 CSV outputs](milestone_5_csv_outputs/). You can open them directly on GitHub. Running the final notebook generates fresh outputs in `results/random/`, as listed below.

| Output in `results/random/` | Contents |
| --- | --- |
| `data_audit.csv` | Yearly counts before and after filtering |
| `validation_search.csv` | Candidate settings and validation ROC-AUC |
| `model_comparison.csv`, `tree_comparison.csv` | Main test metrics |
| `test_roc.png` | Test ROC curves for kNN and logistic regression |
| `threshold_comparison.csv`, `loan_purpose_errors.csv` | Cutoff comparison and denial recall by loan purpose |
| `bagged_tree_seed_runs.csv` | Test scores and training times for the five bagged-tree refits |
| `bootstrap_test_predictions.csv` | Saved test predictions used for paired resampling |
| `bootstrap_test_summary.csv`, `bootstrap_paired_auc_differences.csv` | Bootstrap intervals for test scores and paired comparisons |

## Limitations

The study uses four input features, one state, and one held-out test year. The models predict historical lender decisions. Default denial recall remained low, and differences across years cannot isolate the effect of the financial crisis.
