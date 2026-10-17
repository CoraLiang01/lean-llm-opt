**Mathematical Model**

Let  

- \(I=\{i\}\) be the set of storage areas (indexed by *StorageID*).  
- \(J=\{j\}\) be the set of air‑conditioner types (indexed by *ProductName*).  

Parameters  

- \(\text{capacity}_i\) – capacity of storage area \(i\) (units of size).  
- \(\text{value}_j\) – monetary value of one unit of product \(j\).  
- \(\text{weight}_j\) – size (weight) of one unit of product \(j\).  

Decision variables  

- \(x_{ij}\in\mathbb{Z}_{\ge 0}\) – number of units of product \(j\) placed in storage area \(i\).

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J}\text{value}_j\,x_{ij} \\[4pt]
\text{s.t.}\;& \sum_{j\in J}\text{weight}_j\,x_{ij}\;\le\;\text{capacity}_i
\quad \forall\,i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0}\quad \forall\,i\in I,\;j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(\text{capacity}_i\) | `file_0_view_0` | `Capacity` |
| \(\text{value}_j\) | `file_1_view_0` | `Value` |
| \(\text{weight}_j\) | `file_1_view_0` | `Weight` |

Index sets are derived from the primary keys of the respective tables (`StorageID` for `file_0_view_0`, `ProductName` for `file_1_view_0`).