**Mathematical Model**

Let  

- \(I\) be the set of all product names in the file.  
- For each \(i\in I\)  
  - \(r_i\)  – revenue per unit (from column **Revenue**)  
  - \(d_i\)  – demand quantity (from column **Demand**)  
  - \(s_i\)  – initial inventory (from column **Initial Inventory**)  

Decision variable  

- \(x_i \ge 0\) – quantity of product \(i\) to be sold.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & x_i \;\le\; d_i &&\forall i\in I \\[2pt]
             & x_i \;\le\; s_i &&\forall i\in I \\[2pt]
             & x_i \;\ge\; 0   &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter / Variable | Source Table | Source Column |
|----------------------|--------------|---------------|
| \(I\) (index set)   | `file_0_view_0` | `Product Name` |
| \(r_i\)              | `file_0_view_0` | `Revenue` |
| \(d_i\)              | `file_0_view_0` | `Demand` |
| \(s_i\)              | `file_0_view_0` | `Initial Inventory` |
| \(x_i\)              | – (decision variable) | – |

All numeric values are taken directly from the corresponding columns; no aggregation or transformation is applied beyond the constraints listed above.