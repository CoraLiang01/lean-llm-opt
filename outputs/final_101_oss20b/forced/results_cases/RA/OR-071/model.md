**Symbolic Model**

Let  

* \(I\) – set of products (indexed by the *Product Name* in *41.csv*).  
* \(x_i\) – production quantity of product \(i\) (continuous, \(x_i\ge 0\)).  

Parameters (all taken from table **file_0_view_0**):

| Symbol | Meaning | Source column |
|--------|---------|---------------|
| \(L_i\) | Labor required per unit of product \(i\) | “Labor per unit” |
| \(M_i\) | Material required per unit of product \(i\) | “Material per unit” |
| \(P_i\) | Selling price per unit of product \(i\) | “Selling Price” |
| \(C_i\) | Variable cost per unit of product \(i\) | “Variable Cost” |

Fixed weekly operating cost: \(F = 4\,500\).

---

**Objective**

\[
\max_{x}\;\; \sum_{i\in I}\bigl(P_i-C_i\bigr)\,x_i \;-\; F
\]

---

**Constraints**

1. **Labor capacity**

\[
\sum_{i\in I} L_i\,x_i \;\le\; 1\,650
\]

2. **Material capacity**

\[
\sum_{i\in I} M_i\,x_i \;\le\; 1\,850
\]

3. **Non‑negativity**

\[
x_i \;\ge\; 0 \quad \forall i\in I
\]

---

**Data Mapping**

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| \(L_i\) | file_0_view_0 | Labor per unit |
| \(M_i\) | file_0_view_0 | Material per unit |
| \(P_i\) | file_0_view_0 | Selling Price |
| \(C_i\) | file_0_view_0 | Variable Cost |
| Product index \(i\) | file_0_view_0 | Product Name |

All other data (fixed cost, capacities) are constants defined in the model.