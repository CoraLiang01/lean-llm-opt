**Mathematical Model**

Let  

* \(I=\{1,\dots ,8\}\) – set of store sections (SectionID).  
* \(J=\{1,\dots ,10\}\) – set of products (ProductName).  

Parameters  

* \(\text{Cap}[i]\) – available display space in section \(i\).  
* \(\text{Val}[j]\) – revenue per unit of product \(j\).  
* \(\text{Wgt}[j]\) – shelf‑space requirement per unit of product \(j\).  

Decision variables  

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; j\in J
\]

Objective  

\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{Val}[j]\;x_{ij}
\]

Capacity constraints (one per section)  

\[
\sum_{j\in J}\text{Wgt}[j]\;x_{ij}\;\le\;\text{Cap}[i]\qquad \forall i\in I
\]

All other constraints are implicit (non‑negativity and integrality of \(x_{ij}\)).  

---

**Data Mapping Rules**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(\text{Cap}[i]\) | `file_0_view_0` | `Capacity` |
| \(\text{Val}[j]\) | `file_1_view_0` | `Value` |
| \(\text{Wgt}[j]\) | `file_1_view_0` | `Weight` |

*Index sets*  
- \(I\) is obtained from the distinct values of `SectionID` in `file_0_view_0`.  
- \(J\) is obtained from the distinct values of `ProductName` in `file_1_view_0`.  

All mappings use the exact identifiers and column names as provided; no aggregation or transformation of data is performed.