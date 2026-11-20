# Dataset documentation

The primary source is the supplied AURA synthetic customer-retention dataset. It is development/demo data and does not establish real-world predictive validity. No outside dataset is used.

- Full data: 12,000 rows and 36 columns.
- Identifier: `customer_id` (excluded from model features).
- Target: `churn` (0/1), with 8,515 non-churned and 3,485 churned customers (29.04%).
- Inputs: the 34 columns marked `Input feature` in `AURA_Data_Dictionary.csv`.
- Supplied train/test split: 9,600 rows (2,788 churned) and 2,400 rows (697 churned).
- Validation: no missing values, duplicate rows, duplicate IDs, invalid target labels, or train/test ID overlap; split rows and target counts reconcile to full data.

See `data/reports/validation.json` for the programmatic result. `data/raw` files are preserved unchanged. The timestamped event history described by AURA is not in these aggregate files; any later event history must be labelled simulated.

Known limitations: synthetic distribution and labels, no external validation, no longitudinal outcomes, and predictive feature associations do not establish causes.
