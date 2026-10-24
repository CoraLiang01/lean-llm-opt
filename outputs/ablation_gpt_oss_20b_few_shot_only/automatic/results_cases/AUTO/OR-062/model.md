**Mathematical Model**

Let  

* \(I\) – set of suppliers (rows of *fixed_cost.csv*).  
* \(J\) – set of stores (rows of *demand.csv*).  

Parameters  

* \(f_i\) – fixed cost of opening supplier \(i\).  
* \(d_j\) – demand of store \(j\).  
* \(c_{ij}\) – unit transportation cost from supplier \(i\) to store \(j\).  
* \(M = \sum_{j\in J} d_j\) – a valid big‑\(M\) bound on total shipments from any supplier.

Decision variables  

* \(y_i \in \{0,1\}\) – 1 if supplier \(i\) is opened, 0 otherwise.  
* \(x_{ij} \ge 0\) – quantity shipped from supplier \(i\) to store \(j\).

---

\[
\begin{aligned}
\min\;& \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{i\in I} x_{ij} = d_j , && \forall j\in J \quad\text{(demand)}\\[4pt]
& \sum_{j\in J} x_{ij} \le M\,y_i , && \forall i\in I \quad\text{(activation)}\\[4pt]
& x_{ij}\ge 0 , && \forall i\in I,\; j\in J\\
& y_i\in\{0,1\} , && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source file | Column(s) used |
|-----------|-------------|----------------|
| \(I\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/fixed_cost.csv` | `Unnamed: 0` (supplier names) |
| \(f_i\) | same file | `fixed_costs` |
| \(J\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/demand.csv` | `Customer_1`, `Customer_2`, … (store names) |
| \(d_j\) | same file | `Customer_1`, `Customer_2`, … (demand values) |
| \(c_{ij}\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv` | `Unnamed: 0` (supplier names) as rows, store columns (`CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO`, `BANCROFT`) as columns |

All indices, parameters, and variables are defined exactly as they appear in the source files; no numerical values are enumerated in the model.