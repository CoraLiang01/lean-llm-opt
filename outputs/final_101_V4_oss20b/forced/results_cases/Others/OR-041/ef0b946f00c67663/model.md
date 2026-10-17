## Mathematical Model

Let  

* \(I=\{1,\dots ,20\}\) be the set of real‑estate areas (index by the rows of **products.csv**).  
* \(v_i\) – benefit per unit of development in area \(i\) (column **Value**).  
* \(w_i\) – resource consumption per unit in area \(i\) (column **Weight**).  
* \(C\) – total development capacity (column **Capacity**).  
* \(x_i\) – scale of development per day in area \(i\) (decision variable).

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \;\ge\; 0 \qquad \forall i\in I
\end{aligned}
\]

---

## Data Mapping

| Parameter | Source | Table ID | Column(s) | Notes |
|-----------|--------|----------|-----------|-------|
| \(C\) | capacity.csv | `file_0_view_0` | `Capacity` | Overall capacity limit |
| \(v_i\) | products.csv | `file_1_view_0` | `Value` | Benefit per unit for area \(i\) |
| \(w_i\) | products.csv | `file_1_view_0` | `Weight` | Resource consumption per unit for area \(i\) |
| \(I\) | products.csv | `file_1_view_0` | `ProductName` | Index set of areas |

All parameters are taken directly from the corresponding columns of the specified tables; no additional data or transformations are introduced. The decision variables \(x_i\) are continuous and non‑negative, as implied by the problem statement.