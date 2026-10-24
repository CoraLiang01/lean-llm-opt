**Mathematical Model**

Let  

- \(I\) be the set of car‑model records returned from the filtered view  
  `file_0_view_0` (each record corresponds to a distinct product variant).  

- For every \(i\in I\):

  - \(r_i\)  – revenue per unit (from column **Revenue**).  
  - \(d_i\)  – demand quantity (from column **Demand**).  
  - \(s_i\)  – initial inventory (from column **Initial Inventory**).  

- Decision variable  
  \[
  x_i \;\ge\; 0 \qquad (i\in I)
  \]
  – quantity of product \(i\) that the dealership plans to sell.

---

### Objective
\[
\max \;\sum_{i\in I} r_i\,x_i
\]

### Constraints
\[
\begin{aligned}
&0 \;\le\; x_i \;\le\; d_i &&\forall i\in I \quad &&\text{(cannot sell more than demand)}\\[4pt]
&0 \;\le\; x_i \;\le\; s_i &&\forall i\in I \quad &&\text{(cannot sell more than inventory)}
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Source Column | Mapping |
|-----------|--------------|---------------|---------|
| \(r_i\)   | `file_0_view_0` | `Revenue` | `r[i] = Revenue` |
| \(d_i\)   | `file_0_view_0` | `Demand` | `d[i] = Demand` |
| \(s_i\)   | `file_0_view_0` | `Initial Inventory` | `s[i] = Initial Inventory` |
| Index \(i\) | `file_0_view_0` | row index (1‑based) | `i = row_index` |

All parameters are taken directly from the filtered view; no additional aggregation or transformation is performed. The model uses continuous decision variables as specified in the query.