**Mathematical Model**

Let  

* \(S\) – set of suppliers (indexed by \(i\)).  
* \(C\) – set of supermarkets (indexed by \(j\)).  

Parameters  

* \(f_i\) – fixed opening cost of supplier \(i\).  
* \(d_j\) – demand of supermarket \(j\).  
* \(t_{ij}\) – transportation cost per unit from supplier \(i\) to supermarket \(j\).  

Decision variables  

* \(y_i \in \{0,1\}\) – 1 if supplier \(i\) is opened, 0 otherwise.  
* \(x_{ij} \ge 0\) – quantity shipped from supplier \(i\) to supermarket \(j\).  

\[
\begin{aligned}
\min\;& \sum_{i\in S} f_i\,y_i \;+\; \sum_{i\in S}\sum_{j\in C} t_{ij}\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{i\in S} x_{ij} = d_j , && \forall j\in C \quad\text{(demand satisfaction)}\\[4pt]
& x_{ij} \le M\,y_i , && \forall i\in S,\; j\in C \quad\text{(implicit capacity, }M\text{ large)}\\[4pt]
& y_i \in \{0,1\}, && \forall i\in S\\
& x_{ij} \ge 0 , && \forall i\in S,\; j\in C
\end{aligned}
\]

*(The second constraint is optional; if suppliers have no capacity limit it can be omitted.)*

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(f_i\) | `file_1_view_0` | `fixed_costs` |
| \(d_j\) | `file_0_view_0` | `demand` |
| \(t_{ij}\) | `file_2_view_0` | `C1` … `C12` (columns named after customers) |

*Index sets*  
* Suppliers \(S\) are identified by the values in column `Unnamed: 0` of `file_1_view_0`.  
* Supermarkets \(C\) are identified by the values in column `customer` of `file_0_view_0`.  

All parameter values are taken directly from the corresponding cells in the tables; no additional data are introduced.