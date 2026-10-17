**Mathematical Model**

Let  

* \(I=\{i\}\) be the set of warehouses (rows of *fixed_cost.csv*).  
* \(J=\{j\}\) be the set of musicians/bands (rows of *demand.csv*).  

Parameters  

* \(f_i\) – fixed cost of activating warehouse \(i\).  
* \(c_{ij}\) – unit transportation cost from warehouse \(i\) to customer \(j\).  
* \(d_j\) – demand of customer \(j\).  
* \(M=\sum_{j\in J} d_j\) – a valid big‑\(M\) upper bound on total shipments from any warehouse.

Decision variables  

* \(x_{ij}\ge 0\) – quantity shipped from warehouse \(i\) to customer \(j\).  
* \(y_i\in\{0,1\}\) – 1 if warehouse \(i\) is activated, 0 otherwise.

\[
\begin{aligned}
\min\;&\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}\;+\;\sum_{i\in I} f_i\,y_i \\[4pt]
\text{s.t. } &\sum_{i\in I} x_{ij}=d_j, &&\forall j\in J \quad\text{(demand)}\\[2pt]
&\sum_{j\in J} x_{ij}\le M\,y_i, &&\forall i\in I \quad\text{(activation)}\\[2pt]
&x_{ij}\ge 0,\; y_i\in\{0,1\}, &&\forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Row ID Column | Value Column |
|-----------|--------------|---------------|--------------|
| \(f_i\)   | `file_1_view_0` | `Unnamed: 0` | `fixed_costs` |
| \(c_{ij}\)| `file_2_view_0` | `Unnamed: 0` | `C1`, `C2`, `C3` (columns correspond to customers) |
| \(d_j\)   | `file_0_view_0` | `customer`   | `demand` |

| Index Set | Source Table | ID Column |
|-----------|--------------|-----------|
| \(I\) | `file_1_view_0` | `Unnamed: 0` |
| \(J\) | `file_0_view_0` | `customer` |

The big‑\(M\) value is computed as \(M=\sum_{j\in J} d_j\) using the `demand` column of `file_0_view_0`.