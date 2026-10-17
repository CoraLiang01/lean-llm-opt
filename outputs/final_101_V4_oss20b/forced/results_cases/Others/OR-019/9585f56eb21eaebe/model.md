## Mathematical Model

Let  

- \(I=\{i\mid\text{Product Name contains “27in”}\}\)  (the two products in the view).  
- \(r_i\) – revenue per unit of product \(i\).  
- \(d_i\) – demand (maximum units that can be sold) of product \(i\).  
- \(s_i\) – initial inventory of product \(i\).  
- \(x_i\) – number of units of product \(i\) to sell (decision variable).

### Objective
\[
\max \;\sum_{i\in I} r_i\,x_i
\]

### Constraints
\[
\begin{aligned}
x_i &\le d_i &&\forall i\in I &&\text{(cannot exceed demand)}\\[4pt]
x_i &\le s_i &&\forall i\in I &&\text{(cannot exceed inventory)}\\[4pt]
x_i &\ge 0 &&\forall i\in I &&\text{(non‑negativity)}\\[4pt]
x_i &\in \mathbb{Z} &&\forall i\in I &&\text{(integer units)}
\end{aligned}
\]

---

## Data Mapping

| Parameter / Variable | Source Table | Column(s) | Notes |
|----------------------|--------------|-----------|-------|
| \(I\) | `file_0_view_0` | `Product Name` | Filter: contains “27in” |
| \(r_i\) | `file_0_view_0` | `Revenue` | Revenue per unit |
| \(d_i\) | `file_0_view_0` | `Demand` | Demand quantity |
| \(s_i\) | `file_0_view_0` | `Initial Inventory` | Initial stock |
| \(x_i\) | Decision variable | – | Integer, \(\ge 0\) |

All parameters are taken directly from the filtered view `file_0_view_0`; no additional data or assumptions are introduced. The model uses only the two products that satisfy the filter condition.