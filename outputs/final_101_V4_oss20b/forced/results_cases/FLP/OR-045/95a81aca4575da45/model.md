**Mathematical Model**

Let  

- \(I\) be the set of product types.  
- \(w_i\) be the weight of one unit of product \(i\).  
- \(v_i\) be the benefit (value) of one unit of product \(i\).  
- \(C\) be the total available capacity.  
- \(x_i\) be the integer number of units of product \(i\) ordered daily.

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(I\) | Product identifiers | `file_1_view_0` | `ProductName` |
| \(w_i\) | Weight of product | `file_1_view_0` | `Weight` |
| \(v_i\) | Value (benefit) of product | `file_1_view_0` | `Value` |
| \(C\) | Total capacity | `file_0_view_0` | `Capacity` |

All parameters are taken directly from the corresponding CSV files; no numerical values are enumerated in the model.