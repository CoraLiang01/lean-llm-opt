**Mathematical Model**

Let  

- \(I=\{i\}\) be the set of cabinets (CabinetID from *capacity.csv*).  
- \(J=\{j\}\) be the set of coffee products (ProductName from *products.csv*).  

Parameters  

- \(c_i\) – capacity of cabinet \(i\) (Capacity).  
- \(v_j\) – value per unit of product \(j\) (Value).  
- \(w_j\) – weight per unit of product \(j\) (Weight).  

Decision variables  

- \(x_{ij}\in \mathbb{Z}_{\ge 0}\) – number of units of product \(j\) placed in cabinet \(i\).

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij} \\[4pt]
\text{s.t.}\;& \sum_{j\in J} w_j\,x_{ij}\;\le\; c_i , &&\forall i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0}, &&\forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(c_i\) | `file_0_view_0` | `Capacity` |
| \(v_j\) | `file_1_view_0` | `Value` |
| \(w_j\) | `file_1_view_0` | `Weight` |

Index sets are derived from the unique values of `CabinetID` and `ProductName` in the respective tables.