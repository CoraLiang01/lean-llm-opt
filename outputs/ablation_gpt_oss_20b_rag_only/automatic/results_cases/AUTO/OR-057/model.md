**Mathematical Model**

Let  

- \(P\) be the set of platforms indexed by the values in column **PlatformID** of *capacity.csv*.  
- \(G\) be the set of games indexed by the values in column **ProductName** of *products.csv*.  

Parameters (all non‑negative real numbers)

- \(c_p\) – memory capacity of platform \(p\)  \(\;(p\in P)\)  
- \(v_g\) – value of game \(g\)  \(\;(g\in G)\)  
- \(w_g\) – memory weight of game \(g\)  \(\;(g\in G)\)

Decision variables

- \(x_{p,g}\in \mathbb{Z}_{\ge 0}\) – number of units of game \(g\) listed on platform \(p\).

---

**Objective**

\[
\max \;\; \sum_{p\in P}\;\sum_{g\in G}\; v_g \, x_{p,g}
\]

---

**Constraints**

1. **Platform capacity**

\[
\forall p\in P:\qquad
\sum_{g\in G}\; w_g \, x_{p,g}\;\le\; c_p
\]

2. **Non‑negativity & integrality**

\[
\forall p\in P,\; \forall g\in G:\qquad
x_{p,g}\in \mathbb{Z}_{\ge 0}
\]

---

**Data Mapping**

| Parameter | Table ID | Column |
|-----------|----------|--------|
| \(c_p\)   | file_0_view_0 | Capacity |
| \(v_g\)   | file_1_view_0 | Value |
| \(w_g\)   | file_1_view_0 | Weight |

| Index Set | Table ID | Column |
|-----------|----------|--------|
| \(P\) | file_0_view_0 | PlatformID |
| \(G\) | file_1_view_0 | ProductName |

All other data (e.g., row identifiers) are ignored. The model uses the exact identifiers and column names from the CSVQA data. No additional constraints or variables are introduced.