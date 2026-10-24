**Mathematical Model**

Let  

- \(S\) be the set of store sections indexed by the values in column **SectionID** of *capacity.csv*.  
- \(P\) be the set of products indexed by the values in column **ProductName** of *products.csv*.  

Parameters  

- \(C_s\) – display‑space capacity of section \(s\) (units of space).  
- \(v_p\) – unit revenue (price) of product \(p\).  
- \(w_p\) – space requirement per unit of product \(p\).  

Decision variables  

- \(x_{s p}\in \mathbb{Z}_{\ge 0}\) – number of units of product \(p\) stocked in section \(s\).  

**Objective**  

\[
\max \;\; \sum_{s\in S}\sum_{p\in P} v_p\,x_{s p}
\]

**Constraints**  

1. **Capacity limits (per section)**  

\[
\sum_{p\in P} w_p\,x_{s p}\;\le\; C_s \qquad \forall s\in S
\]

2. **Non‑negativity and integrality**  

\[
x_{s p}\in \mathbb{Z}_{\ge 0}\qquad \forall s\in S,\;p\in P
\]

---

**Data Mapping**

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| \(C_s\)   | file_0_view_0 | Capacity |
| \(v_p\)   | file_1_view_0 | Value |
| \(w_p\)   | file_1_view_0 | Weight |

All other data (index sets \(S\) and \(P\)) are derived from the primary keys **SectionID** and **ProductName** in the respective tables. No additional constraints or parameters are introduced beyond those specified in the query.