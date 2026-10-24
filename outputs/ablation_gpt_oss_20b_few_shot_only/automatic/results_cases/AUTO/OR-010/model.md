**Mathematical Model**

Let  

- \(I\) be the set of all products (indexed by the *Product Name* column).  
- For each \(i\in I\)  
  - \(r_i\) = revenue per unit (from column **Revenue**)  
  - \(d_i\) = deterministic demand (from column **Demand**)  
  - \(s_i\) = initial inventory (from column **Initial Inventory**)  

Decision variable  

- \(x_i \in \mathbb{Z}_{\ge 0}\) = number of units of product \(i\) that are fulfilled.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& 0 \le x_i \le d_i && \forall i\in I \\[2pt]
& 0 \le x_i \le s_i && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source File | Column Name |
|--------|-------------|-------------|
| \(I\) | `Test_Dataset/Large-scale-or/NRM_testing/NRM1/MobileSalesDataset.csv` | `Product Name` |
| \(r_i\) | same file | `Revenue` |
| \(d_i\) | same file | `Demand` |
| \(s_i\) | same file | `Initial Inventory` |
| \(x_i\) | decision variable | – (integer, non‑negative) |

All parameters are taken directly from the specified CSV columns; no additional data are introduced.