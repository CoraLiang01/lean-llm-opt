**Mathematical Model**

Let  

- \(I=\{i\mid i\text{ is a ShelfID in }{\tt capacity.csv}\}\)  
- \(J=\{j\mid j\text{ is a ProductName in }{\tt products.csv}\}\)

Parameters  

- \(C_i\) – capacity of shelf \(i\) (from column **Capacity** of **file_0_view_0**)  
- \(V_j\) – value of product \(j\) (from column **Value** of **file_1_view_0**)  
- \(W_j\) – weight of product \(j\) (from column **Weight** of **file_1_view_0**)

Decision variables  

- \(x_{ij}\in\mathbb{Z}_{\ge 0}\) – number of units of product \(j\) placed on shelf \(i\)

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J} V_j\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{j\in J} W_j\,x_{ij}\;\le\; C_i , && \forall i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0}, && \forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(C_i\)   | `file_0_view_0` | `Capacity` |
| \(V_j\)   | `file_1_view_0` | `Value` |
| \(W_j\)   | `file_1_view_0` | `Weight` |

All index sets \(I\) and \(J\) are taken from the complete lists of `ShelfID` and `ProductName` values in the respective tables. No additional constraints or bounds are imposed beyond those stated.