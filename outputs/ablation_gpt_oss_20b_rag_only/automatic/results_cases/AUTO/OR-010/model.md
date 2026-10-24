**Mathematical Model**

Let  

- \(I\) be the set of products (indexed by the value in the column **“Product Name”**).  
- For each \(i\in I\)  
  - \(r_i\)  – revenue per unit (parameter)  
  - \(d_i\)  – deterministic demand (parameter)  
  - \(s_i\)  – initial inventory (parameter)  
  - \(x_i\)  – quantity of product \(i\) to fulfill (decision variable)

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le d_i && \forall i\in I \\[2pt]
             & 0 \le x_i \le s_i && \forall i\in I \\[2pt]
             & x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| \(r_i\)   | file_0_view_0 | Revenue |
| \(d_i\)   | file_0_view_0 | Demand |
| \(s_i\)   | file_0_view_0 | Initial Inventory |

The index set \(I\) is derived from the distinct values in the column **“Product Name”** of the same table. No other tables or columns are used.