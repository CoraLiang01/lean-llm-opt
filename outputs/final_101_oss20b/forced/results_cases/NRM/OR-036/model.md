**Mathematical Model**

Let  

- \(I\) – set of vehicle types (indexed by the rows of *products.csv*).  
- \(C\) – total inventory capacity (from *capacity.csv*).  

Parameters (all taken directly from the source tables)

| Symbol | Source | Column |
|--------|--------|--------|
| \(C\) | file_0_view_0 | Capacity |
| \(v_i\) | file_1_view_0 | Value |
| \(w_i\) | file_1_view_0 | Weight |

Decision variables  

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

Objective  

\[
\max \;\; \sum_{i\in I} v_i\,x_i
\]

Capacity constraint  

\[
\sum_{i\in I} w_i\,x_i \;\le\; C
\]

Non‑negativity (implicit in the integer domain)  

\[
x_i \ge 0 \qquad \forall i\in I
\]

---

**Data Mapping**

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| \(C\) | file_0_view_0 | Capacity |
| \(v_i\) | file_1_view_0 | Value |
| \(w_i\) | file_1_view_0 | Weight |

All parameter values are taken from the corresponding rows of the indicated tables; no additional data or enumeration is required.