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
| \(F\) | Fixed weekly operating cost | constant \(4500\) |

**Objective**

\[
\max_{x}\;\; \sum_{i\in I}\bigl(P_i-C_i\bigr)\,x_i \;-\; F
\]

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

**Data Mapping Rules**

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| \(L_i\) | file_0_view_0 | Labor per unit |
| \(M_i\) | file_0_view_0 | Material per unit |
| \(P_i\) | file_0_view_0 | Selling Price |
| \(C_i\) | file_0_view_0 | Variable Cost |
| \(x_i\) | – | – (decision variable) |
| \(F\) | – | – (constant 4500) |

The index set \(I\) is derived from all distinct *Product Name* values in **file_0_view_0**. No values are enumerated; the model remains symbolic and ready for numerical instantiation.