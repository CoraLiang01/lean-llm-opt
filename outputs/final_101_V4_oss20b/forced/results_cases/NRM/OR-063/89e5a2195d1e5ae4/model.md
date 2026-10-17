**Symbolic Model**

\[
\begin{aligned}
\text{Minimize}\quad & \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{subject to}\quad
& \sum_{i\in I} x_{ij} \;=\; d_j , &&\forall j\in J \qquad\text{(demand)}\\[4pt]
& x_{ij} \;\le\; d_j\,y_i , &&\forall i\in I,\; j\in J \qquad\text{(warehouse activation)}\\[4pt]
& x_{ij}\;\ge\;0 , &&\forall i\in I,\; j\in J \\[4pt]
& y_i\in\{0,1\}, &&\forall i\in I
\end{aligned}
\]

**Index Sets**

- \(I\): set of warehouses (rows of *fixed_cost.csv* and *transportation_costs.csv*).  
- \(J\): set of customers (rows of *demand.csv* and columns of *transportation_costs.csv*).

**Parameters**

- \(f_i\): fixed operating cost of warehouse \(i\).  
- \(d_j\): demand of customer \(j\).  
- \(c_{ij}\): transportation cost per unit from warehouse \(i\) to customer \(j\).

**Variables**

- \(y_i\in\{0,1\}\): 1 if warehouse \(i\) is activated, 0 otherwise.  
- \(x_{ij}\ge 0\): quantity supplied from warehouse \(i\) to customer \(j\).

---

### Data Mapping

| Parameter | Source Table | Source Column | Mapping Detail |
|-----------|--------------|---------------|----------------|
| \(f_i\) | `file_1_view_0` | `fixed_costs` | Row identifier `Unnamed: 0` → warehouse index \(i\) |
| \(d_j\) | `file_0_view_0` | `demand` | Row identifier `customer` → customer index \(j\) |
| \(c_{ij}\) | `file_2_view_0` | `C1`, `C2`, `C3`, `C4` | Row identifier `Unnamed: 0` → warehouse \(i\); column name (`C1`–`C4`) → customer \(j\) |

No numerical values are listed; the model references the parameters symbolically and ties each to its exact source table and column.