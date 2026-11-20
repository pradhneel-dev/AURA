# Model training summary

Selected **xgboost** by highest stratified training-only cross-validation ROC-AUC. Candidate comparison metrics use the fixed 0.50 threshold. All candidates were selected with stratified cross-validation on the 9,600-row training split; the 2,400-row supplied test split was used once for final reporting. Recall-first ranking supports retention outreach but may create more false positives; review precision before using actions. The operational decision threshold is 0.38; selected using training-only out-of-fold predictions with a 20% stated contact-capacity limit. Held-out results at that threshold are in threshold_evaluation.json. Feature importance describes fitted-model associations, not causes. SHAP is not generated; the selected estimator's coefficient magnitude or impurity importance is reported as an alternative.

```
              model  accuracy  precision  recall     f1  roc_auc  pr_auc  threshold  validation_roc_auc
logistic_regression    0.5863     0.3647  0.5725 0.4456   0.6206  0.4053     0.5000              0.6171
      random_forest    0.7046     0.4709  0.1392 0.2148   0.6279  0.3953     0.5000              0.6140
            xgboost    0.7108     0.5108  0.1019 0.1699   0.6475  0.4181     0.5000              0.6278
```
