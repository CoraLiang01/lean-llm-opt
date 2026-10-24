**Mathematical Model**

Let  

- \(I\) be the set of bread types (identified by *ProductName*).  
- \(v_i\) be the expected profit per unit of type \(i\).  
- \(w_i\) be the storage weight per unit of type \(i\).  
- \(C\) be the daily storage capacity.  
- \(x_i\) be the number of units of type \(i\) ordered today.

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
| \(C\) | Storage capacity | `file_0_view_0` | `Capacity` |
| \(v_i\) | Profit per unit | `file_1_view_0` | `Value` |
| \(w_i\) | Weight per unit | `file_1_view_0` | `Weight` |
| \(I\) | Bread types | `file_1_view_0` | `ProductName` |

All parameters are taken directly from the corresponding CSV files; no numerical values are enumerated in the model.