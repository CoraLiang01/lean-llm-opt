**Mathematical Model**

Let  

- \(I=\{i\mid i\text{ is a SectionID in } \text{capacity.csv}\}\)  
- \(J=\{j\mid j\text{ is a ProductName in } \text{products.csv}\}\)

Parameters  

- \(\text{capacity}[i]\) – display space limit of section \(i\)  
- \(\text{value}[j]\) – price (revenue per unit) of product \(j\)  
- \(\text{weight}[j]\) – shelf‑space requirement per unit of product \(j\)

Decision variables  

- \(x_{ij}\in \mathbb{Z}_{+}\) – number of units of product \(j\) placed in section \(i\)

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J}\text{value}[j]\;x_{ij} \\[4pt]
\text{s.t. } & \sum_{j\in J}\text{weight}[j]\;x_{ij}\;\le\;\text{capacity}[i],
\qquad \forall\, i\in I \\[4pt]
& x_{ij}\in \mathbb{Z}_{+},\qquad \forall\, i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Key Column | Value Column |
|-----------|--------------|------------|--------------|
| \(\text{capacity}[i]\) | `file_0_view_0` | `SectionID` | `Capacity` |
| \(\text{value}[j]\) | `file_1_view_0` | `ProductName` | `Value` |
| \(\text{weight}[j]\) | `file_1_view_0` | `ProductName` | `Weight` |

| Variable | Definition |
|----------|------------|
| \(x_{ij}\) | Integer decision variable for units of product \(j\) in section \(i\) |

Sets \(I\) and \(J\) are the distinct values of `SectionID` and `ProductName` respectively, taken from the full returned rows of the corresponding tables. No additional constraints or parameters are introduced.