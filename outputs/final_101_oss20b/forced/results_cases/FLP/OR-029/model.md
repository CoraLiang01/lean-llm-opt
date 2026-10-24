**Mathematical Model**

Let  

- \(I\) be the set of FAUX products (one element per returned row).  
- \(x_i\) – number of units of product \(i\) to be sold.  

Parameters (to be instantiated from the data)

- \(R_i\) – revenue per unit of product \(i\).  
- \(D_i\) – demand quantity of product \(i\).  
- \(I_i\) – initial inventory of product \(i\).

\[
\begin{aligned}
\max\;& \sum_{i\in I} R_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le D_i && \forall i\in I \\[2pt]
             & 0 \le x_i \le I_i && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(I\) | Index set | `file_0_view_0` | *Product Name* (filtered where contains “FAUX”) |
| \(R_i\) | Revenue | `file_0_view_0` | `Revenue` |
| \(D_i\) | Demand | `file_0_view_0` | `Demand` |
| \(I_i\) | Initial Inventory | `file_0_view_0` | `Initial Inventory` |

All parameters are taken directly from the corresponding columns of the filtered view `file_0_view_0`; no numerical values are enumerated in the model.