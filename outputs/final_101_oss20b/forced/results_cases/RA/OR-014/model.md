**Mathematical Model**

Let  

- \(I\) be the set of pizza types (indexed by the values in the column *Product Name*).  
- \(r_i\) – revenue per unit of pizza type \(i\) (column *Revenue*).  
- \(d_i\) – deterministic demand for pizza type \(i\) (column *Demand*).  
- \(inv_i\) – initial inventory of pizza type \(i\) (column *Initial Inventory*).  

Decision variable  

- \(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of pizza type \(i\) to be sold.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& x_i \;\le\; d_i &&\forall i\in I \\[2pt]
& x_i \;\le\; inv_i &&\forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{\ge 0} &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(r_i\) | `file_0_view_0` | `Revenue` |
| \(d_i\) | `file_0_view_0` | `Demand` |
| \(inv_i\) | `file_0_view_0` | `Initial Inventory` |
| Index set \(I\) | `file_0_view_0` | `Product Name` |

All parameters are taken directly from the corresponding columns of the single source table `file_0_view_0`. No values are enumerated in the model; the solver will read the data via the mapping above.