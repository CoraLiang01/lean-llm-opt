**Mathematical Model**

Let  

- \(P\) be the set of product identifiers (one element per row in *products.csv*).  
- For each \(p\in P\):  
  - \(w_p\) = weight of one unit of product \(p\).  
  - \(v_p\) = benefit (value) of one unit of product \(p\).  
- \(C\) = overall inventory capacity (total weight that can be ordered).  

Decision variables  

- \(x_p \in \mathbb{Z}_{\ge 0}\) – number of units of product \(p\) to order daily.

**Objective**

\[
\max \;\; \sum_{p\in P} v_p\,x_p
\]

**Capacity constraint**

\[
\sum_{p\in P} w_p\,x_p \;\le\; C
\]

---

**Data Mapping**

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| \(C\) | `file_0_view_0` | `Capacity` |
| \(w_p\) | `file_1_view_0` | `Weight` |
| \(v_p\) | `file_1_view_0` | `Value` |

*All tables are used in full; no rows are omitted or sorted.*