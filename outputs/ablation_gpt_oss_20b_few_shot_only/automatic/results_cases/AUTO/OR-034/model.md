**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \text{Revenue}_i \; x_i \\[4pt]
\text{subject to}\quad
& 0 \;\le\; x_i \;\le\; \text{Demand}_i , && \forall i\in I \\[2pt]
& 0 \;\le\; x_i \;\le\; \text{Inventory}_i , && \forall i\in I \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}, && \forall i\in I
\end{aligned}
\]

*Indices*  
\(I\) – set of all baked‑good products.

*Parameters*  
\(\text{Revenue}_i\) – revenue per unit of product \(i\).  
\(\text{Demand}_i\) – deterministic demand for product \(i\).  
\(\text{Inventory}_i\) – initial inventory of product \(i\).

*Decision variables*  
\(x_i\) – number of units of product \(i\) to be sold (fulfilled).

---

**Data Mapping**

| Parameter | Source File | Column Name |
|-----------|-------------|-------------|
| \(I\) (product identifiers) | `Test_Dataset/Large-scale-or/NRM_testing/NRM25/Frenchbakerydailysales.csv` | `Product Name` |
| \(\text{Revenue}_i\) | same file | `Revenue` |
| \(\text{Demand}_i\) | same file | `Demand` |
| \(\text{Inventory}_i\) | same file | `Initial Inventory` |

All parameters are read directly from the specified CSV; no values are enumerated in the model. The variable domain and constraints follow exactly the problem statement.