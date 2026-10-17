**Mathematical Model**

Let  

- \(I\) be the set of cabinet identifiers (from *capacity.csv*).  
- \(J\) be the set of product names (from *products.csv*).  

Parameters  

- \(c_i\) – capacity of cabinet \(i\).  
- \(v_j\) – value of one unit of product \(j\).  
- \(w_j\) – weight of one unit of product \(j\).  

Decision variables  

- \(x_{ij}\) – integer number of units of product \(j\) placed in cabinet \(i\).  

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij} \\[4pt]
\text{s.t.}\;& \sum_{j\in J} w_j\,x_{ij} \;\le\; c_i \qquad \forall i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{+}\qquad \forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(c_i\)   | `file_0_view_0` | `Capacity` |
| \(v_j\)   | `file_1_view_0` | `Value` |
| \(w_j\)   | `file_1_view_0` | `Weight` |

Index sets are derived from the primary keys of the respective tables:

- \(I = \{ \text{CabinetID} \}\) from `file_0_view_0`.
- \(J = \{ \text{ProductName} \}\) from `file_1_view_0`.