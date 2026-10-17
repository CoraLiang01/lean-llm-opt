**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \text{rev}_i\,x_i \\[4pt]
\text{subject to}\quad
& x_i \;\le\; \text{dem}_i && \forall i\in I \\[2pt]
& x_i \;\le\; \text{inv}_i && \forall i\in I \\[2pt]
& x_i \;\ge\; 0 && \forall i\in I
\end{aligned}
\]

*Indices*  
\(I\) – set of tablet models (one element per row of the filtered table).

*Parameters*  
\(\text{rev}_i\) – revenue per unit of model \(i\).  
\(\text{dem}_i\) – deterministic demand for model \(i\).  
\(\text{inv}_i\) – initial inventory of model \(i\).

*Decision variable*  
\(x_i\) – number of units of model \(i\) to be sold (fulfilled).

---

**Data Mapping Rules**

| Parameter | Source Table ID | Source Column |
|-----------|-----------------|---------------|
| `rev_i`   | `file_0_view_0` | `Revenue`     |
| `dem_i`   | `file_0_view_0` | `Demand`      |
| `inv_i`   | `file_0_view_0` | `Initial Inventory` |
| Index `i` | `file_0_view_0` | `Product Name` (used to identify each tablet model) |

All parameters are taken directly from the corresponding columns of the filtered table `file_0_view_0`. No additional data or transformations are introduced.