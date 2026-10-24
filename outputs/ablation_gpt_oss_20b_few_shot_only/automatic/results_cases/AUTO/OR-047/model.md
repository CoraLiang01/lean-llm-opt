**Mathematical Model**

Let  

- \(I=\{i\mid i\text{ is a PlatformId in } \texttt{capacity.csv}\}\)  
- \(J=\{j\mid j\text{ is a ProductName in } \texttt{products.csv}\}\)

Parameters  

- \(\text{Capacity}[i]\) – memory capacity of platform \(i\)  
- \(\text{Value}[j]\) – value of genre \(j\)  
- \(\text{Weight}[j]\) – memory requirement of genre \(j\)

Decision variables  

- \(x_{ij}\in \mathbb{Z}_{\ge 0}\) – number of units of genre \(j\) listed on platform \(i\)

\[
\begin{aligned}
\max\;& \sum_{i\in I}\;\sum_{j\in J}\; \text{Value}[j]\;x_{ij} \\[4pt]
\text{s.t.}\;& \sum_{j\in J}\; \text{Weight}[j]\;x_{ij}\;\le\;\text{Capacity}[i],
\qquad \forall\, i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0},\qquad \forall\, i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | File Path | Column(s) |
|-----------|-----------|-----------|
| \(\text{Capacity}[i]\) | `Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv` | `PlatformId`, `Capacity` |
| \(\text{Value}[j]\) | `Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv` | `ProductName`, `Value` |
| \(\text{Weight}[j]\) | `Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv` | `ProductName`, `Weight` |

*Index sets \(I\) and \(J\) are derived from the unique values of `PlatformId` and `ProductName` respectively.*