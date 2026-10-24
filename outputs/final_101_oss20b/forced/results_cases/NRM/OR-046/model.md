**Mathematical Model**

Let  

- \(I\) be the set of products.  
- For each \(i\in I\)  
  - \(w_i\) = weight of one unit of product \(i\) (from *products.csv*).  
  - \(v_i\) = value (benefit) of one unit of product \(i\) (from *products.csv*).  
- \(C\) = overall stock capacity (from *capacity.csv*).  
- \(x_i\) = number of units of product \(i\) to order each day.

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(C\) | `file_0_view_0` | `Capacity` |
| \(w_i\) | `file_1_view_0` | `Weight` |
| \(v_i\) | `file_1_view_0` | `Value` |
| Index set \(I\) | `file_1_view_0` | `ProductName` |

All parameter values are taken directly from the corresponding columns of the specified tables; no additional data or transformations are introduced.