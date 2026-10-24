**Mathematical Model**

Let  

- \(P\) be the set of all products in *41.csv* (index \(p\)).  
- Decision variable \(x_p\) = production quantity of product \(p\) (continuous, \(x_p\ge 0\)).  

Parameters (all obtained from *file_0_view_0*)  

| Symbol | Meaning | Source column |
|--------|---------|---------------|
| \(l_p\) | Labor required per unit of product \(p\) | “Labor per unit” |
| \(m_p\) | Material required per unit of product \(p\) | “Material per unit” |
| \(s_p\) | Selling price per unit of product \(p\) | “Selling Price” |
| \(v_p\) | Variable cost per unit of product \(p\) | “Variable Cost” |

Fixed weekly operating cost: \(F = 4500\).

---

**Objective**

\[
\max_{x}\; \sum_{p\in P} (s_p - v_p)\,x_p \;-\; F
\]

---

**Constraints**

1. **Labor capacity**

\[
\sum_{p\in P} l_p\,x_p \;\le\; 1650
\]

2. **Material capacity**

\[
\sum_{p\in P} m_p\,x_p \;\le\; 1850
\]

3. **Non‑negativity**

\[
x_p \;\ge\; 0 \quad \forall p\in P
\]

---

**Data Mapping**

| Table ID | Column in CSV | Parameter |
|----------|---------------|-----------|
| file_0_view_0 | “Product Name” | index \(p\) |
| file_0_view_0 | “Labor per unit” | \(l_p\) |
| file_0_view_0 | “Material per unit” | \(m_p\) |
| file_0_view_0 | “Selling Price” | \(s_p\) |
| file_0_view_0 | “Variable Cost” | \(v_p\) |

All parameter values are taken directly from the corresponding rows of *41.csv*; no enumeration of individual values is shown.