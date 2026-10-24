**Mathematical Model**

Let  

- \(I\) be the set of product names appearing in *file_0_view_0* (the two 27‑inch products).  
- For each \(i\in I\)  
  - \(Revenue_i\)  – revenue per unit of product \(i\).  
  - \(Demand_i\)   – deterministic demand for product \(i\).  
  - \(Inv_i\)      – initial inventory of product \(i\).  

Decision variable  

- \(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of product \(i\) to be sold.

\[
\begin{aligned}
\max\;& \sum_{i\in I} Revenue_i \, x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le Demand_i && \forall i\in I \\[2pt]
             & 0 \le x_i \le Inv_i      && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Table ID | Column Name |
|--------|----------|-------------|
| \(I\) | file_0_view_0 | Product Name |
| \(Revenue_i\) | file_0_view_0 | Revenue |
| \(Demand_i\) | file_0_view_0 | Demand |
| \(Inv_i\) | file_0_view_0 | Initial Inventory |
| \(x_i\) | – | – (decision variable, integer, non‑negative) |

All parameters are taken directly from the corresponding columns of *file_0_view_0*; no additional data are introduced.