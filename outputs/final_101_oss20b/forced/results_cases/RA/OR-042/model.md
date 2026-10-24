**Mathematical Model**

Let  

- \(I\) be the set of product types (rows of *products.csv*).  
- \(v_i\) be the benefit coefficient (column **Value**) for product \(i\).  
- \(w_i\) be the weight (column **Weight**) for product \(i\).  
- \(C\) be the overall inventory capacity (column **Capacity** from *capacity.csv*).  
- \(x_i\) be the daily order quantity of product \(i\).

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
| \(v_i\) | `file_1_view_0` | `Value` |
| \(w_i\) | `file_1_view_0` | `Weight` |
| Index set \(I\) | `file_1_view_0` | `ProductName` |

All parameters are taken directly from the corresponding CSV files; no values are enumerated in the model.