**Mathematical Model**

Let  

- \(I=\{i\mid\text{Product Name of }i\text{ begins with }S700_\}\)  
- \(R_i\) – revenue per unit of product \(i\)  
- \(D_i\) – deterministic demand for product \(i\) over the horizon  
- \(I_i\) – initial inventory of product \(i\)  

Decision variable  

- \(x_i\in\mathbb{Z}_{+}\) – number of units of product \(i\) the store intends to sell  

\[
\begin{aligned}
\max\;&\sum_{i\in I} R_i\,x_i \\[4pt]
\text{s.t.}\;&x_i \le D_i &&\forall i\in I \\[2pt]
&x_i \le I_i &&\forall i\in I \\[2pt]
&x_i \ge 0 &&\forall i\in I
\end{aligned}
\]

(Variables are implicitly integer because \(x_i\in\mathbb{Z}_{+}\).)

---

**Data Mapping**

| Parameter | Source File | Source Column |
|-----------|-------------|---------------|
| \(R_i\) | `Test_Dataset/Large-scale-or/NRM_testing/NRM15/SampleSalesData.csv` | `Revenue` |
| \(D_i\) | `Test_Dataset/Large-scale-or/NRM_testing/NRM15/SampleSalesData.csv` | `Demand` |
| \(I_i\) | `Test_Dataset/Large-scale-or/NRM_testing/NRM15/SampleSalesData.csv` | `Initial Inventory` |

The index set \(I\) is derived from the same file by selecting rows whose `Product Name` starts with the prefix `S700_`. No other data or constraints are introduced beyond those explicitly stated in the query.