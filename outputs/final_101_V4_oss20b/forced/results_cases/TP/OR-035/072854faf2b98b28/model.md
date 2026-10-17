**Mathematical Model**

Let  

* \(I\) – set of product types (indexed by product name).  
* \(p_i\) – expected profit per unit of product \(i\).  
* \(w_i\) – storage space (weight) per unit of product \(i\).  
* \(C\) – total available storage capacity.  
* \(x_i\) – number of units of product \(i\) to order (integer, \(x_i\ge 0\)).

\[
\begin{aligned}
\max\;& \sum_{i\in I} p_i\,x_i \\[4pt]
\text{s.t. } & \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Column(s) | Notes |
|-----------|--------------|-----------|-------|
| \(C\) | `file_0_view_0` | `Capacity` | Single scalar value |
| \(p_i\) | `file_1_view_0` | `Value` | Indexed by `ProductName` |
| \(w_i\) | `file_1_view_0` | `Weight` | Indexed by `ProductName` |
| \(I\) | `file_1_view_0` | `ProductName` | Set of all product names |

All numeric values are taken directly from the corresponding CSV files; no additional aggregation or transformation is performed.