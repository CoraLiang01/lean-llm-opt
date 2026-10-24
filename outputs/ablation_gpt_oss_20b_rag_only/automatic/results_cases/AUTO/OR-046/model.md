**Mathematical Model**

Let  

- \(P\) be the set of products indexed by the value of **ProductName** in *products.csv*.  
- \(x_p\) be the number of units of product \(p\) ordered each day.  

Parameters  

- \(v_p\) – profit (value) per unit of product \(p\).  
- \(w_p\) – weight (space) per unit of product \(p\).  
- \(C\) – total stock‑capacity available.  

**Objective**  
\[
\max \;\sum_{p\in P} v_p\,x_p
\]

**Constraints**  

1. **Capacity**  
\[
\sum_{p\in P} w_p\,x_p \;\le\; C
\]

2. **Non‑negativity & integrality**  
\[
x_p \;\in\; \mathbb{Z}_{\ge 0}\quad \forall p\in P
\]

---

**Data Mapping**

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| \(C\) | `file_0_view_0` | `Capacity` |
| \(v_p\) | `file_1_view_0` | `Value` |
| \(w_p\) | `file_1_view_0` | `Weight` |
| Index set \(P\) | `file_1_view_0` | `ProductName` |

All records from the returned tables are used; no additional data are introduced.