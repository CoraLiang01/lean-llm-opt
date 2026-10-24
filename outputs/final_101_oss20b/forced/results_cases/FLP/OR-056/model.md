**Mathematical Model**

Let  

- \(I=\{i\}\) be the set of display areas (DisplayID).  
- \(J=\{j\}\) be the set of vessel types (ProductName).  

Parameters  

- \(\text{capacity}_i\) – capacity of display area \(i\).  
- \(\text{value}_j\) – monetary value of vessel type \(j\).  
- \(\text{weight}_j\) – size (weight) of vessel type \(j\).  

Decision variables  

- \(x_{ij}\in \mathbb{Z}_{+}\) – number of vessels of type \(j\) placed in display area \(i\).

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J}\text{value}_j\,x_{ij} \\[4pt]
\text{s.t.}\;& \sum_{j\in J}\text{weight}_j\,x_{ij}\;\le\;\text{capacity}_i,
\qquad \forall i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{+},\qquad \forall i\in I,\;j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(\text{capacity}_i\) | `file_0_view_0` | `Capacity` |
| \(\text{value}_j\) | `file_1_view_0` | `Value` |
| \(\text{weight}_j\) | `file_1_view_0` | `Weight` |

Index sets are derived from the unique values of `DisplayID` in `file_0_view_0` and `ProductName` in `file_1_view_0`.