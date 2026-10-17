**Mathematical Model**

Let  

- \(I\) be the set of shelves indexed by the values of **ShelfID** in *file_0_view_0*.  
- \(J\) be the set of products indexed by the values of **ProductName** in *file_1_view_0*.  

Parameters  

- \(\text{Cap}_i\) – capacity of shelf \(i\) (column **Capacity** in *file_0_view_0*).  
- \(\text{Val}_j\) – value of product \(j\) (column **Value** in *file_1_view_0*).  
- \(\text{Wgt}_j\) – weight of product \(j\) (column **Weight** in *file_1_view_0*).  

Decision variables  

- \(x_{ij}\in \mathbb{Z}_{\ge 0}\) – number of units of product \(j\) placed on shelf \(i\).

\[
\begin{aligned}
\max\;& \sum_{i\in I}\;\sum_{j\in J}\; \text{Val}_j \, x_{ij} \\[4pt]
\text{s.t.}\;& \sum_{j\in J}\; \text{Wgt}_j \, x_{ij}\;\le\; \text{Cap}_i,
\qquad \forall\, i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0},
\qquad \forall\, i\in I,\; j\in J .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Table ID | Column |
|-----------|----------|--------|
| \(\text{Cap}_i\) | `file_0_view_0` | `Capacity` |
| \(\text{Val}_j\) | `file_1_view_0` | `Value` |
| \(\text{Wgt}_j\) | `file_1_view_0` | `Weight` |
| Index \(I\) | `file_0_view_0` | `ShelfID` |
| Index \(J\) | `file_1_view_0` | `ProductName` |

All indices, parameters, and variables are defined solely from the supplied tables; no additional data are introduced.