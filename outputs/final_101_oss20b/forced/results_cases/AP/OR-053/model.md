**Mathematical Model**

Let  

- \(I\) be the set of shelves (indexed by `ShelfID` in *capacity.csv*).  
- \(J\) be the set of products (indexed by `ProductName` in *products.csv*).  

Parameters  

- \(c_i\) – capacity of shelf \(i\) (units of weight).  
- \(v_j\) – value of one unit of product \(j\).  
- \(w_j\) – weight of one unit of product \(j\).  

Decision variables  

- \(x_{ij}\in \mathbb{Z}_{\ge 0}\) – number of units of product \(j\) placed on shelf \(i\).

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{j\in J} w_j\,x_{ij} \;\le\; c_i \qquad \forall i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(c_i\)   | `file_0_view_0` | `Capacity` (indexed by `ShelfID`) |
| \(v_j\)   | `file_1_view_0` | `Value` (indexed by `ProductName`) |
| \(w_j\)   | `file_1_view_0` | `Weight` (indexed by `ProductName`) |

No additional constraints or bounds are required beyond those stated.