**Mathematical Model**

Let  

- \(I\) be the set of dairy products (indexed by the values in the column **Full_Product_Name**).  
- For each \(i\in I\)  
  - \(rev_i\)  – revenue per unit (column **Revenue**)  
  - \(dem_i\)  – demand quantity (column **Demand**)  
  - \(inv_i\)  – initial inventory (column **Initial Inventory**)  

Decision variable  

- \(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of product \(i\) to fulfill.

\[
\begin{aligned}
\max\;& \sum_{i\in I} rev_i \, x_i \\[4pt]
\text{s.t.}\;& 0 \le x_i \le dem_i && \forall i\in I \\[2pt]
& 0 \le x_i \le inv_i && \forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | File Path | Column Name |
|-----------|-----------|-------------|
| \(rev_i\) | `Test_Dataset/Large-scale-or/NRM_testing/NRM22/DairyGoodsSalesDataset.csv` | `Revenue` |
| \(dem_i\) | `Test_Dataset/Large-scale-or/NRM_testing/NRM22/DairyGoodsSalesDataset.csv` | `Demand` |
| \(inv_i\) | `Test_Dataset/Large-scale-or/NRM_testing/NRM22/DairyGoodsSalesDataset.csv` | `Initial Inventory` |
| Index set \(I\) | `Test_Dataset/Large-scale-or/NRM_testing/NRM22/DairyGoodsSalesDataset.csv` | `Full_Product_Name` |

All parameters are taken directly from the specified columns; no additional data are introduced.