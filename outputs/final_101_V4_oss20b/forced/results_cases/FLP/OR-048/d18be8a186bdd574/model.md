**Mathematical Model**

Let  

- \(I=\{i\}\) be the set of storage areas (indexed by *StorageID*).  
- \(J=\{j\}\) be the set of air‑conditioner types (indexed by *ProductName*).  

Parameters  

- \(c_i\) – capacity of storage area \(i\) (units of size).  
- \(v_j\) – value of one unit of product \(j\).  
- \(w_j\) – size (weight) of one unit of product \(j\).  

Decision variables  

- \(x_{ij}\in \mathbb{Z}_{\ge 0}\) – number of units of product \(j\) placed in storage area \(i\).

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{j\in J} w_j\,x_{ij} \;\le\; c_i, \qquad \forall i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0}, \qquad \forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(c_i\)   | `file_0_view_0` | `Capacity` |
| \(v_j\)   | `file_1_view_0` | `Value` |
| \(w_j\)   | `file_1_view_0` | `Weight` |

Indices are derived from the primary identifiers in each table:

- \(i\) from `StorageID` in `file_0_view_0`.
- \(j\) from `ProductName` in `file_1_view_0`.