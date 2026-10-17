**Mathematical model**

Let  

* \(I\) – set of all products (index taken from the rows of *41.csv*).  
* \(x_i\) – production quantity of product \(i\) (continuous, \(x_i\ge 0\)).  

Parameters (all taken from *file_0_view_0*)  

| Symbol | Meaning | Source column |
|--------|---------|---------------|
| \(L_i\) | labor required per unit of product \(i\) | “Labor per unit” |
| \(M_i\) | material required per unit of product \(i\) | “Material per unit” |
| \(P_i\) | selling price per unit of product \(i\) | “Selling Price” |
| \(C_i\) | variable cost per unit of product \(i\) | “Variable Cost” |
| \(F\) | fixed weekly operating cost | constant \(4500\) |
| \(L^{\max}\) | weekly labor capacity | constant \(1650\) |
| \(M^{\max}\) | weekly material capacity | constant \(1850\) |

The objective is to maximize weekly net profit:

\[
\max_{x}\;\; \sum_{i\in I}\bigl(P_i-C_i\bigr)\,x_i \;-\; F
\]

Subject to the resource limits

\[
\begin{aligned}
\sum_{i\in I} L_i\,x_i &\le L^{\max} \quad &&\text{(labor)}\\[4pt]
\sum_{i\in I} M_i\,x_i &\le M^{\max} \quad &&\text{(material)}\\[4pt]
x_i &\ge 0 \quad &&\forall i\in I
\end{aligned}
\]

All variables \(x_i\) are continuous (non‑negative).

---

**Data‑mapping rules**

| Parameter | Table ID | Column name | Notes |
|-----------|----------|-------------|-------|
| \(L_i\) | file_0_view_0 | Labor per unit | float |
| \(M_i\) | file_0_view_0 | Material per unit | float |
| \(P_i\) | file_0_view_0 | Selling Price | float |
| \(C_i\) | file_0_view_0 | Variable Cost | float |
| \(F\) | – | – | constant 4500 |
| \(L^{\max}\) | – | – | constant 1650 |
| \(M^{\max}\) | – | – | constant 1850 |
| Index set \(I\) | file_0_view_0 | Product Name | each distinct product name |

The objective coefficient for product \(i\) is \(P_i-C_i\) (derived from the two columns).  
The RHS of the labor constraint is \(L^{\max}\); the RHS of the material constraint is \(M^{\max}\).  
All decision variables \(x_i\) are indexed by the product names from the same table.