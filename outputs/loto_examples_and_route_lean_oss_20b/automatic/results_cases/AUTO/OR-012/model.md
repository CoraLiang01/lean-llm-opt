**Mathematical Model**

Let  

- \(I\) be the set of products (identified by *Product Name*).  
- For each \(i\in I\)  
  - \(p_i\)  – revenue per unit (column **Revenue**)  
  - \(d_i\)  – deterministic demand (column **Demand**)  
  - \(s_i\)  – initial inventory (column **Initial Inventory**)  
- Decision variable \(x_i\) – number of units of product \(i\) to sell.

\[
\begin{aligned}
\max\;& \sum_{i\in I} p_i\,x_i \\[4pt]
\text{s.t.}\;& 0 \le x_i \le d_i && \forall i\in I \\[2pt]
& 0 \le x_i \le s_i && \forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Table ID | Column Name |
|--------|----------|-------------|
| \(I\) | *file_0_view_0* | *Product Name* |
| \(p_i\) | *file_0_view_0* | *Revenue* |
| \(d_i\) | *file_0_view_0* | *Demand* |
| \(s_i\) | *file_0_view_0* | *Initial Inventory* |

All parameters are taken directly from the corresponding columns of the CSV view *file_0_view_0*. No values are enumerated; the model remains symbolic and ready for numerical instantiation.