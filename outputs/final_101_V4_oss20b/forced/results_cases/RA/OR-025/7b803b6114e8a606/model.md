**Mathematical Model**

Let  

- \(I\) be the set of tablet product identifiers (the *Product Name* values in the filtered view).  

- For each \(i\in I\)  
  - \(rev_i\)  – revenue per unit (column **Revenue**)  
  - \(dem_i\)  – demand quantity (column **Demand**)  
  - \(inv_i\)  – initial inventory (column **Initial Inventory**)  

- Decision variable  
  - \(x_i\in\mathbb{Z}_{\ge 0}\) – number of units of product \(i\) to sell.

---

\[
\begin{aligned}
\max\;& \sum_{i\in I} rev_i \, x_i \\[4pt]
\text{s.t.}\;& x_i \;\le\; dem_i && \forall i\in I \\[2pt]
& x_i \;\le\; inv_i && \forall i\in I \\[2pt]
& x_i \;\ge\; 0 && \forall i\in I \\[2pt]
& x_i \in \mathbb{Z} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source Table | Column |
|--------|--------------|--------|
| \(I\) | `file_0_view_0` | *Product Name* (each row defines one element of \(I\)) |
| \(rev_i\) | `file_0_view_0` | *Revenue* |
| \(dem_i\) | `file_0_view_0` | *Demand* |
| \(inv_i\) | `file_0_view_0` | *Initial Inventory* |

All parameters are taken directly from the corresponding columns of the filtered view `file_0_view_0`. No additional data or assumptions are introduced.