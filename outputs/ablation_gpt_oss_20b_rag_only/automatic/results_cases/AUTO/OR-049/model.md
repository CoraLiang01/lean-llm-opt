**Mathematical Model**

Let  

- \(S\) be the set of shelf identifiers (from *capacity.csv*).  
- \(P\) be the set of product names (from *products.csv*).  

Parameters  

- \(\text{Cap}[s]\) – capacity of shelf \(s\).  
- \(\text{Val}[p]\) – monetary value of one unit of product \(p\).  
- \(\text{Wgt}[p]\) – weight of one unit of product \(p\).  

Decision variables  

- \(x_{s p}\in \mathbb{Z}_{\ge 0}\) – number of units of product \(p\) placed on shelf \(s\).  

**Objective**  

\[
\max \;\; \sum_{s\in S}\;\sum_{p\in P}\; \text{Val}[p]\;x_{s p}
\]

**Constraints**  

1. Shelf‑capacity (weight) limits  

\[
\sum_{p\in P}\; \text{Wgt}[p]\;x_{s p}\;\le\;\text{Cap}[s] \qquad \forall s\in S
\]

2. Non‑negativity and integrality  

\[
x_{s p}\;\in\;\mathbb{Z}_{\ge 0}\qquad \forall s\in S,\;p\in P
\]

---

**Data Mapping (CSVQA_DATA)**  

| Symbol | Table ID | Column | Index Key |
|--------|----------|--------|-----------|
| \(S\) | `file_0_view_0` | `ShelfID` | – |
| \(\text{Cap}[s]\) | `file_0_view_0` | `Capacity` | `ShelfID` |
| \(P\) | `file_1_view_0` | `ProductName` | – |
| \(\text{Val}[p]\) | `file_1_view_0` | `Value` | `ProductName` |
| \(\text{Wgt}[p]\) | `file_1_view_0` | `Weight` | `ProductName` |

All other columns in the source files are ignored. The model uses the exact identifiers and numeric values supplied in the observation.