**Mathematical Model**

Let  

- \(I\) be the set of product types (indexed by the *ProductName* values).  
- For each \(i\in I\)  
  - \(p_i\)  – profit (value) of product \(i\)  \(\;(=\) *Value*).  
  - \(w_i\)  – weight (inventory consumption) of product \(i\)  \(\;(=\) *Weight*).  
- \(C\) – total inventory capacity  \(\;(=\) *Capacity*).  

Decision variable  

- \(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of product \(i\) to order per day.

\[
\begin{aligned}
\max\;& \sum_{i\in I} p_i\,x_i \\[4pt]
\text{s.t. } & \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \;\ge\; 0,\; x_i \in \mathbb{Z}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column Name |
|-----------|--------|----------|-------------|
| \(C\) | Capacity | `file_0_view_0` | `Capacity` |
| \(p_i\) | Profit per product | `file_1_view_0` | `Value` |
| \(w_i\) | Weight per product | `file_1_view_0` | `Weight` |
| \(I\) | Product identifiers | `file_1_view_0` | `ProductName` |

All parameter values are taken directly from the corresponding columns in the specified tables; no additional data or transformations are introduced.