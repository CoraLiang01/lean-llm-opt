**Mathematical Model**

Let  

- \(I=\{\,i\mid i\text{ is a ShelfID in } \text{capacity.csv}\,\}\)  
- \(J=\{\,j\mid j\text{ is a ProductName in } \text{products.csv}\,\}\)

Parameters  

- \(C_i\) – capacity of shelf \(i\)  
- \(V_j\) – value per unit of product \(j\)  
- \(W_j\) – weight per unit of product \(j\)

Decision variables  

- \(x_{ij}\in \mathbb{Z}_{+}\) – number of units of product \(j\) placed on shelf \(i\)

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J} V_j\,x_{ij} \\[4pt]
\text{s.t.}\;& \sum_{j\in J} W_j\,x_{ij}\;\le\; C_i , \qquad \forall i\in I \\[4pt]
& x_{ij}\;\ge\;0 ,\; x_{ij}\in \mathbb{Z}, \qquad \forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(C_i\)   | `file_0_view_0` | `Capacity` |
| \(V_j\)   | `file_1_view_0` | `Value` |
| \(W_j\)   | `file_1_view_0` | `Weight` |
| Index \(I\) | `file_0_view_0` | `ShelfID` |
| Index \(J\) | `file_1_view_0` | `ProductName` |

All data values are taken directly from the corresponding tables; no additional data or constraints are introduced.