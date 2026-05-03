# IEEE-CIS Fraud Detection - XGBoost

### კონკურსის მიმოხილვა

Kaggle IEEE-CIS Fraud Detection კონკურსის მიზანია საბანკო ტრანზაქციებში თაღლითობის გამოვლენა ბარათის, მოწყობილობის, ელ-ფოსტისა და ტრანზაქციის მახასიათებლების საფუძველზე. ამოცანა ფასდება ROC-AUC მეტრიკით.

---

## რეპოზიტორიის სტრუქტურა

```
IEEE-CIS-Fraud-Detection/
│
├── model_experiment_XGBoost.ipynb  ← EDA, cleaning, feature engineering, feature selection, ექსპერიმენტები
├── model_experiment_DecisionTree.ipynb  ← EDA, cleaning, feature engineering, feature selection, ექსპერიმენტები
├── model_experiment_RandomForest.ipynb  ← EDA, cleaning, feature engineering, feature selection, ექსპერიმენტები
├── model_inference.ipynb           ← საუკეთესო მოდელის ჩატვირთვა Model Registry-იდან, submission
└── README.md
```

---

## ფაილების აღწერა

| ფაილი | აღწერა |
|---|---|
| `model_experiment_XGBoost.ipynb` | XGBoost notebook — data cleaning, feature engineering, feature selection და მოდელების ექსპერიმენტები |
| `model_experiment_DecisionTree.ipynb` | DecisionTree notebook — data cleaning, feature engineering, feature selection და მოდელების ექსპერიმენტები |
| `model_experiment_RandomForest.ipynb` | RandomForest notebook — data cleaning, feature engineering, feature selection და მოდელების ექსპერიმენტები |
| `model_inference.ipynb` | MLflow Model Registry-იდან საუკეთესო pipeline-ის ჩატვირთვა, Kaggle-ის ტესტ სეტზე პროგნოზი და submission.csv გენერაცია |

---

## DecisionTree და RandomForest მოდელები გამოვიყენე საცდელად, რადგან ამხელა დატას, თანაც არაბალანსირებულს ინტუიციურად ყველაზე კარგად XGBoost ამოხსნიდა. რადგან ორივე ხის ალგორითმია, პრეპროცესინგი ორივეგან ერთნაირია. RandomForest-მა უკეთესი შედეგი აჩვენა, ვიდრე DecisionTree, რაც არაა გასაკვირი. XGBoost-მა ორივე დაააუთპერფორმა, ამიტომ დეტალურად მხოლოდ მას განვიხილავ(პრეპროცესინგიც სხვანაირია).

## EDA — მონაცემების ანალიზი

### 1. კლასების განაწილება

მონაცემები არაა ბალანსირებული, ანუ fraud ტრანზაქციები მთლიანი მონაცემების მხოლოდ ~3.5%-ს შეადგენს.

![class_distribution](images/class_distribution.png)

ამ პრობლემის გადასაჭრელად გამოვიყენე `scale_pos_weight=27.58`.

### 2. TransactionAmt განაწილება

Raw TransactionAmt skewed-ია. log1p ტრანსფორმაციის შემდეგ განაწილება უფრო ნორმალური ხდება.

![transaction_amt](images/transaction_amt.png)

### 3. TransactionAmt — Legit vs Fraud

Fraud ტრანზაქციები განსხვავებულ განაწილებას აჩვენებს legit-თან შედარებით, კერძოდ fraud-ებს მეტად გაფანტული თანხები აქვთ.

![amt_legit_fraud](images/amt_legit_fraud.png)

### 4. დროის პატერნები

Transaction Hour და Day ანალიზი გვიჩვენებს, რომ fraud ტრანზაქციები ღამით (0-5 სთ) ოდნავ მაღალი სიხშირით ხდება.

![time_patterns](images/time_patterns.png)

### 5. Fraud Rate კატეგორიული სვეტებით

- **ProductCD=C** ყველაზე მაღალი fraud rate-ით (~12%)
- **card4=discover** ყველაზე სარისკო ბარათის ტიპი (~8%)
- **card6=credit** გაცილებით სარისკოა debit-ზე (~7% vs ~2.5%)
- **P_emaildomain=protonmail.com** ყველაზე მაღალი fraud rate-ით (~40%)

![fraud_rates](images/fraud_rates.png)

### 6. Missing Values

მრავალი სვეტი 85%+ missing values-ს შეიცავს.

![missing_values](images/missing_values.png)

---

## Data Cleaning

### Missing Values

90%-ზე მეტი missing value-ის მქონე სვეტები ამოვიღე. დარჩენილი სვეტებისთვის:
- **რიცხვითი სვეტები** — შევავსე **მედიანით** 
- **კატეგორიული სვეტები** — შევავსე **მოდით**

fill value-ები გამოვითვალე მხოლოდ train-ზე და გამოვიყენე val/test-ზეც — data leakage-ის თავიდან ასაცილებლად.

---

## Feature Engineering

### ახალი features

| Feature | აღწერა |
|---|---|
| `TransactionAmt_log1p` | log(1+x) ტრანსფორმაცია — skewness შევამცირე |
| `TransactionAmt_cents` | fraud-ებს ხშირად მრგვალი თანხები აქვთ |
| `TransactionDT_day` | ტრანზაქციის დღე |
| `TransactionDT_hour` | ტრანზაქციის საათი |
| `P/R_emaildomain_provider` | ელ-ფოსტის პროვაიდერი |
| `card1-5_amt_mean/std` | card-ის საშუალო და სტდ ტრანზაქციის თანხა |
| `card1_amt_deviation` | ტრანზაქციის გადახრა  |
| `card1_amt_zscore` | z-score card-ის საშუალოსა და სტდ-ის მიხედვით |
| `card1_freq` | card-ის გამოჩენის სიხშირე |
| `card1_txn_count` | card-ის ტრანზაქციების რაოდენობა |
| `card1_unique_amounts` | card-ის უნიკალური თანხების რაოდენობა |
| `card1_time_gap` | დრო წინა ტრანზაქციამდე |
| `card1_unique_items` | card-ის უნიკალური პროდუქტების რაოდენობა |
| `P_email_high_fraud` | protonmail/mail.com |
| `is_night` | ღამის ტრანზაქცია (0-5 სთ) |

### UID Features — მომხმარებლის იდენტიფიკაცია

Dataset-ში მომხმარებლის ID არ არსებობს. uid-ები შეიქმნა სვეტების კომბინაციით:

| UID | კომბინაცია | IV (amt_mean) |
|---|---|---|
| `uid1` | card1 + addr1 | 0.3341 |
| `uid2` | card1 + addr1 + P_emaildomain | 0.2626 |
| `uid3` | card1 + card5 | 0.3958 |

თითოეული uid-ისთვის გამოვითვალე: `amt_mean`, `amt_std`, `txn_count`, `freq` — train-ის სტატისტიკის საფუძველზე.

> ყველა aggregate feature გამოითვლება **train-ზე** და **map**-ით გადაეცემა val/test-ს data leakage-ის თავიდან ასაცილებლად.

---

## კატეგორიული ცვლადების Encoding

### OneHotEncoding (OHE)
დაბალი კარდინალობის სვეტებისთვის: `ProductCD`, `card4`, `card6`, `DeviceType`, `M1`-`M9`

### WOE Encoding (Weight of Evidence)
მაღალი კარდინალობის სვეტებისთვის: `P_emaildomain`, `R_emaildomain`, `DeviceInfo`

---

## Feature Selection

### 1. კორელაციის ფილტრი (Threshold = 0.9)

მაღლად კორელირებული სვეტები ამოიღება. engineered features დაცულია კორელაციის ფილტრისგან, ვამოწმებ მხოლოდ ორიგინალ სვეტებს.

სიჩქარის გაზრდისთვის კორელაციის გამოთვლა ხდება **50,000 სტრიქონის sample-ზე**.

### 2. IV ფილტრი (Threshold = 0.02)

IV < 0.02-ის მქონე სვეტები ამოვშალე.

**საუკეთესო IV მქონე features:**

| Feature | IV |
|---|---|
| V258 | 0.736 |
| V246 | 0.692 |
| card2_amt_mean | 0.575 |
| card1_amt_mean | 0.398 |
| uid3_amt_mean | 0.396 |
| uid3_amt_std | 0.375 |
| uid1_amt_mean | 0.334 |

საბოლოოდ დარჩა 161 feature კორელაციისა და IV ფილტრის შემდეგ.

---

## Pipeline არქიტექტურა

```
Raw Test Data
      ↓
FullPreprocessor     ← missing values, _add_features (aggregates from train), OHE, WOE
      ↓
CorrelationDropper   ← კორელირებული სვეტების ამოღება
      ↓
IVFeatureSelector    ← IV < 0.02 სვეტების ამოღება
      ↓
StandardScaler       ← სტანდარტიზაცია
      ↓
XGBClassifier        ← scale_pos_weight=27.58
      ↓
P(isFraud)
```
---

## ტრენინგი და ექსპერიმენტები

ყველა ექსპერიმენტი დავლოგე MLflow-ში DagsHub-ზე. 5-fold Stratified Cross-Validation train-ზე.

**class imbalance** — `scale_pos_weight = (legit count) / (fraud count) = 27.58`

### საუკეთესო შედეგები

| Run | Train AUC | Val AUC | Test AUC | Overfit Gap |
|---|---|---|---|---|
| n600_depth7_lr0.04_ss0.85 | 0.9895 | **0.9562** | **0.9591** | 0.0333 |
| n300_depth8_lr0.05_ss0.8 | 0.9882 | 0.9543 | 0.9577 | 0.0339 |
| n400_depth6_lr0.05_ss0.8 | 0.9724 | 0.9456 | 0.9476 | 0.0268 |
| n300_depth6_lr0.05_ss0.8 | 0.9622 | 0.9390 | 0.9397 | 0.0232 |
| n200_depth6_lr0.1_ss1.0 | 0.9666 | 0.9412 | 0.9427 | 0.0254 |

### საუკეთესო პარამეტრები

```python
{
    'n_estimators':     600,
    'max_depth':        7,
    'learning_rate':    0.04,
    'subsample':        0.85,
    'colsample_bytree': 0.85,
    'reg_alpha':        0.05,
    'reg_lambda':       1
}
```

---

## Overfitting ანალიზი

Overfit gap მნიშვნელოვანია, მაგალითად საუკეთესო პარამეტრების overfit gap = 0.033, მაგრამ Score: 0.916125 და Private score: 0.871207. Private score უფრო მეტ უნახავ დატაზე ითვლება, ამიტომ ეს ქულა მოსალოდნელია. მაღალი სიღრმის და ესტიმატორების გამო მოდელი ოვერფიტში წავიდა. გატესტილი მოდელებიდან ყველაზე კარგი Public score ამ პარამეტრებმა აჩვენა, მაგრამ Private score-სთვის ავირჩიე ისეთი პარამეტრები, რომელსაც overfit gap შედარებით პატარა ჰქონდა, რათა დიდ უნახავ დატაზე ამხელა სხვაობა არ მოეცა.

```python
{
   'n_estimators':      250,
    'max_depth':        5,
    'learning_rate':    0.07,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'reg_alpha':        0.1,
    'reg_lambda':       1
}
```

ეს პარამეტრები შედარებით ნაკლებად აგრესულია. Overfit gap = 0.018. ვალიდაციაზე 0.93 ქულა აიღო, 0.02-ით ნაკლები ვიდრე წინაზე განხილულმა მოდელმა, მაგრამ Private score-ს მაქსიმიზაციისთვის ეს მოდელი უკეთესია დაბალი overfit gap-ის გამო.
---

## MLflow ექსპერიმენტები DagsHub-ზე

ყველა run დარეგისტრირებულია: [dagshub.com/sansi23/IEEE-CIS-Fraud-Detection](https://dagshub.com/sansi23/IEEE-CIS-Fraud-Detection.mlflow)

XGBoost-ზე დალოგილია საკმაოდ ბევრი run, რადგან ვცადე არაერთი feature selection და feature engineering. ეტაპობრივად ჰიპერპარამეტრებსაც ვცვლიდი.

თითოეულ run-ში დაილოგა:
- ყველა ჰიპერპარამეტრი
- Train / Val / Test AUC
- Feature selection მეტრიკები
- დატრენინგებული pipeline

საუკეთესო pipeline დარეგისტრირდა **Model Registry**-ში სახელით `XGBoost_FraudDetection_Pipeline`.

---

## Kaggle შედეგები


---

## გამოცდილება

- **Feature leakage** — train-ის სტატისტიკა უნდა შეინახოს და map-ით გადაეცეს სხვა dataset-ებს
- **Index alignment** — `sort_values` DataFrame-ში index-ს არ ცვლის, მაგრამ y-თან შეუსაბამობა იწვევს 0.5 AUC-ს (რენდომს). 
- **Pipeline** — პირველად მქონდა pipeline-თან მუშაობის გამოცდილება და დავინახე, თუ როგორ ამარტივებს პროცესს.
- **GPU** — ასწრაფების მიზნით XGBoost-ში tree building გავაკეთე GPU-ზე.
- **Feature Engineering** — საკმაოდ ბევრი დატა ჩარტი ვნახე და ანალიზის შედეგად მოვიფიქრე feature-ები, რომლებიც კარგი მოდელის შექმნაში დამეხმარა.