**Mathematical Model**

Let  

- \(I\) be the set of products indexed by the *Sub Category* values.  
- \(R_i\) – revenue per unit of product \(i\).  
- \(D_i\) – deterministic demand for product \(i\).  
- \(S_i\) – initial inventory of product \(i\).  

Decision variable  

- \(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of product \(i\) to be sold.

\[
\begin{aligned}
\max\;& \sum_{i\in I} R_i\,x_i \\[4pt]
\text{s.t.}\;& 0 \le x_i \le D_i &&\forall i\in I \\[2pt]
& 0 \le x_i \le S_i &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source File | Source Column |
|-----------|-------------|---------------|
| \(I\) (index) | `Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv` | `Sub Category` |
| \(R_i\) | same file | `Revenue` |
| \(D_i\) | same file | `Demand` |
| \(S_i\) | same file | `Initial Inventory` |

All parameter values are taken directly from the corresponding columns of the CSV file; no additional data are introduced.