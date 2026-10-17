**Mathematical Model**

Let  

- \(I\) be the set of all produce types (one element per row of *products.csv*).  
- \(C\) be the overall inventory‑capacity (the single value in *capacity.csv*).  
- \(w_i\) be the weight of one unit of produce type \(i\).  
- \(v_i\) be the benefit (value) of one unit of produce type \(i\).  
- \(x_i\) be the daily order quantity of produce type \(i\).

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(C\) | `file_0_view_0` | `Capacity` |
| \(w_i\) | `file_1_view_0` | `Weight` |
| \(v_i\) | `file_1_view_0` | `Value` |
| Index set \(I\) | `file_1_view_0` | `ProductName` (each distinct row) |

All parameters are taken directly from the corresponding CSV views; no values are hard‑coded in the model.