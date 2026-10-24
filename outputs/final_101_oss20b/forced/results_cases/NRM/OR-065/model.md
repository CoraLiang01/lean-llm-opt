**Mathematical Model**

Let  

* \(I=\{1,\dots,|I|\}\) – set of warehouses (indexed by the rows of *fixed_cost.csv*).  
* \(J=\{1,\dots,|J|\}\) – set of customers (indexed by the rows of *demand.csv*).  

Parameters  

* \(f_i\) – fixed cost of opening warehouse \(i\) (from *fixed_cost.csv*).  
* \(d_j\) – demand of customer \(j\) (from *demand.csv*).  
* \(c_{ij}\) – transportation cost per unit from warehouse \(i\) to customer \(j\) (from *transportation_costs.csv*).  
* \(M\) – a sufficiently large constant, e.g. \(M=\sum_{j\in J} d_j\).

Decision variables  

* \(y_i \in \{0,1\}\) – 1 if warehouse \(i\) is activated, 0 otherwise.  
* \(x_{ij}\ge 0\) – quantity shipped from warehouse \(i\) to customer \(j\).

\[
\begin{aligned}
\min\;& \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t.}\;& \sum_{i\in I} x_{ij} \;\ge\; d_j, \qquad \forall j\in J \quad\text{(demand satisfaction)}\\[4pt]
& x_{ij} \;\le\; M\,y_i, \qquad \forall i\in I,\; j\in J \quad\text{(no shipping if warehouse closed)}\\[4pt]
& y_i \in \{0,1\}, \qquad \forall i\in I\\
& x_{ij}\ge 0, \qquad \forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(f_i\) | `file_1_view_0` | `fixed_costs` |
| \(d_j\) | `file_0_view_0` | `demand` |
| \(c_{ij}\) | `file_2_view_0` | `C1`, `C2`, `C3` (columns corresponding to customers) |

*Index mapping*  
* Warehouse index \(i\) corresponds to the row identifier in `file_1_view_0` (`Unnamed: 0`).  
* Customer index \(j\) corresponds to the row identifier in `file_0_view_0` (`customer`).  
* The matrix `file_2_view_0` aligns rows (warehouses) and columns (customers) exactly with the above indices.