**Mathematical Model**

Let  

- \(I\) be the set of FAUX products (identified by their product names).  
- For each \(i\in I\)  
  - \(R_i\) = revenue per unit of product \(i\).  
  - \(D_i\) = deterministic demand for product \(i\).  
  - \(S_i\) = initial inventory of product \(i\).  

Decision variable  

- \(x_i \ge 0\) : number of units of product \(i\) to be sold (fulfilled).

\[
\begin{aligned}
\max\;& \sum_{i\in I} R_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le D_i && \forall i\in I \\[2pt]
             & 0 \le x_i \le S_i && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(I\) | Index set | *derived from* | *product names in returned rows* |
| \(R_i\) | Revenue | `file_0_view_0` | `Revenue` |
| \(D_i\) | Demand | `file_0_view_0` | `Demand` |
| \(S_i\) | Initial Inventory | `file_0_view_0` | `Initial Inventory` |

*All parameter values are taken directly from the corresponding columns of the filtered rows in `file_0_view_0`.*