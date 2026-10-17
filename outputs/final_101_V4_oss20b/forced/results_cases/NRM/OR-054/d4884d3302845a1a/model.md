**Mathematical Model**

Let  

- \(I\) be the set of shelves (identified by *ShelfID*).  
- \(J\) be the set of products (identified by *ProductName*).  

Parameters  

- \(C_i\) – capacity of shelf \(i\) (units of weight).  
- \(W_j\) – weight of one unit of product \(j\).  
- \(V_j\) – value of one unit of product \(j\).  

Decision variables  

- \(x_{ij}\in \mathbb{Z}_{\ge 0}\) – number of units of product \(j\) placed on shelf \(i\).

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J} V_j\,x_{ij} \\[4pt]
\text{s.t.}\;& \sum_{j\in J} W_j\,x_{ij}\;\le\; C_i , \qquad \forall i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0}, \qquad \forall i\in I,\; j\in J .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(C_i\) | Shelf capacity | `file_0_view_0` | `Capacity` |
| \(W_j\) | Product weight | `file_1_view_0` | `Weight` |
| \(V_j\) | Product value | `file_1_view_0` | `Value` |
| Index \(i\) | Shelf identifier | `file_0_view_0` | `ShelfID` |
| Index \(j\) | Product identifier | `file_1_view_0` | `ProductName` |

All numerical values are taken directly from the corresponding CSV files; no additional data or constraints are introduced beyond those specified in the query.