**Mathematical Model**

Let  

- \(I\) be the set of all product names in *41.csv*  
- \(x_i\) = production quantity of product \(i\)  \(\;(i\in I)\)

Parameters (all taken directly from the CSV columns)

| Symbol | Meaning | Source column |
|--------|---------|---------------|
| \(L_i\) | Labor required per unit of product \(i\) | “Labor per unit” |
| \(M_i\) | Material required per unit of product \(i\) | “Material per unit” |
| \(P_i\) | Selling price per unit of product \(i\) | “Selling Price” |
| \(C_i\) | Variable cost per unit of product \(i\) | “Variable Cost” |

Fixed weekly operating cost: \(F = 4500\).

---

**Objective**

\[
\max_{x}\; \sum_{i\in I} (P_i - C_i)\,x_i \;-\; F
\]

---

**Constraints**

1. **Labor capacity**

\[
\sum_{i\in I} L_i\,x_i \;\le\; 1650
\]

2. **Material capacity**

\[
\sum_{i\in I} M_i\,x_i \;\le\; 1850
\]

3. **Non‑negativity**

\[
x_i \;\ge\; 0 \quad \forall i\in I
\]

---

**Data Mapping**

| Parameter | Table ID | Column name |
|-----------|----------|-------------|
| \(L_i\) | `file_0_view_0` | `Labor per unit` |
| \(M_i\) | `file_0_view_0` | `Material per unit` |
| \(P_i\) | `file_0_view_0` | `Selling Price` |
| \(C_i\) | `file_0_view_0` | `Variable Cost` |

The index set \(I\) is the set of all distinct values in the `Product Name` column of the same table. No other data sources are used.