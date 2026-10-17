**Mathematical Model**

Let  

- \(I\) be the set of cabinets (indices from the *capacity* table).  
- \(J\) be the set of coffee products (indices from the *products* table).  

Parameters  

- \(c_i\) – capacity of cabinet \(i\) (units of weight).  
- \(v_j\) – value of one unit of product \(j\).  
- \(w_j\) – weight of one unit of product \(j\).  

Decision variables  

- \(x_{ij}\in \mathbb{Z}_{\ge 0}\) – number of units of product \(j\) placed in cabinet \(i\).

\[
\begin{aligned}
\max\quad & \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij} \\[4pt]
\text{s.t.}\quad & \sum_{j\in J} w_j\,x_{ij} \;\le\; c_i, \qquad \forall i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0}, \qquad \forall i\in I,\; j\in J .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column(s) |
|-----------|--------------|-----------|
| \(c_i\) | `file_0_view_0` | `CabinetID`, `Capacity` |
| \(v_j\) | `file_1_view_0` | `ProductName`, `Value` |
| \(w_j\) | `file_1_view_0` | `ProductName`, `Weight` |

The index sets \(I\) and \(J\) are derived from the distinct values of `CabinetID` and `ProductName` in the respective tables.