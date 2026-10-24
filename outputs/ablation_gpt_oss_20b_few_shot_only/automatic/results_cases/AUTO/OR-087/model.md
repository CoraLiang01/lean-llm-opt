**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad
&\sum_{p\in P}\sum_{d\in D}\bigl(SellingPrice_p-ProdCost_p\bigr)\,x_{p,d}
 \;-\;\sum_{p\in P}ActivationCost_p\,y_p \\[4pt]
\text{subject to}\quad
&x_{p,d}\;\le\;ProdQuota_p\,y_p , &&\forall p\in P,\;\forall d\in D \quad\text{(daily quota)}\\[2pt]
&\sum_{d\in D}x_{p,d}\;\le\;MaxDemand_p , &&\forall p\in P \quad\text{(demand)}\\[2pt]
&\sum_{d\in D}x_{p,d}\;\ge\;MinBatch_p\,y_p , &&\forall p\in P \quad\text{(minimum batch)}\\[2pt]
&x_{p,d}\in\mathbb{Z}_{+}, &&\forall p\in P,\;\forall d\in D\\[2pt]
&y_p\in\{0,1\}, &&\forall p\in P
\end{aligned}
\]

**Index Sets**

- \(P=\{1,2,\dots,80\}\) (product indices corresponding to columns A1–A80)  
- \(D=\{1,2,\dots,22\}\) (days in the month)

**Parameters**

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(MaxDemand_p\) | Maximum demand (100 kg units) | 36‑1.csv, column “Maximum Demand (100 kg units)” |
| \(SellingPrice_p\) | Selling price ($/100 kg) | 36‑1.csv, column “Selling Price ($/100 kg)” |
| \(ProdCost_p\) | Production cost ($/100 kg) | 36‑1.csv, column “Production Cost ($/100 kg)” |
| \(ProdQuota_p\) | Daily production quota (100 kg units) | 36‑1.csv, column “Production Quota (max per day)” |
| \(ActivationCost_p\) | Fixed activation cost ($) | 36‑2.csv, column “Activation Cost ($)” |
| \(MinBatch_p\) | Minimum batch size (100 kg units) | 36‑3.csv, column “Minimum Batch Size (100 kg units)” |

**Data Mapping**

- **File:** `Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv`  
  - Column “Maximum Demand (100 kg units)” → \(MaxDemand_p\)  
  - Column “Selling Price ($/100 kg)” → \(SellingPrice_p\)  
  - Column “Production Cost ($/100 kg)” → \(ProdCost_p\)  
  - Column “Production Quota (max per day)” → \(ProdQuota_p\)

- **File:** `Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv`  
  - Column “Activation Cost ($)” → \(ActivationCost_p\)

- **File:** `Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv`  
  - Column “Minimum Batch Size (100 kg units)” → \(MinBatch_p\)

The product indices \(p\) correspond to the order of columns A1–A80 in the CSV files. No numerical values are included in the model; all parameters are mapped directly to the specified source columns.