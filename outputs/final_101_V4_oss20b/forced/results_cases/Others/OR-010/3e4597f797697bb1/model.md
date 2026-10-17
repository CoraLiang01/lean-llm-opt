## Mathematical Model

Let  

* \(I\) – set of all products in **MobileSalesDataset.csv** (index \(i\)).  
* \(r_i\) – revenue per unit of product \(i\) (RMB).  
* \(d_i\) – deterministic demand of product \(i\) (units).  
* \(s_i\) – initial inventory of product \(i\) (units).  
* \(x_i\) – number of units of product \(i\) that are actually sold (decision variable).

### Decision Variables
\[
x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

### Objective
\[
\max \;\; \sum_{i\in I} r_i\,x_i
\]

### Constraints
\[
\begin{aligned}
x_i &\le d_i &&\forall i\in I &&\text{(cannot sell more than demand)}\\[4pt]
x_i &\le s_i &&\forall i\in I &&\text{(cannot sell more than available inventory)}
\end{aligned}
\]

No other constraints are required.

---

## Data Mapping

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(I\) | Product identifiers | `file_0_view_0` | `Product Name` |
| \(r_i\) | Revenue per unit | `file_0_view_0` | `Revenue` |
| \(d_i\) | Demand | `file_0_view_0` | `Demand` |
| \(s_i\) | Initial inventory | `file_0_view_0` | `Initial Inventory` |

All parameters are taken directly from the corresponding columns of the single source table `file_0_view_0`. No additional data or transformations are introduced.