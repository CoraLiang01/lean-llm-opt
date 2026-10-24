**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & Z = \sum_{p\in P}\bigl(S_p-C_p\bigr)\,x_p \;-\; F \\[4pt]
\text{subject to}\quad
& \sum_{p\in P} L_p\,x_p \;\le\; 1650 &&\text{(labor capacity)}\\
& \sum_{p\in P} M_p\,x_p \;\le\; 1850 &&\text{(material capacity)}\\
& x_p \;\ge\; 0 &&\forall\,p\in P
\end{aligned}
\]

**Index set**

- \(P\) – set of all product names listed in *41.csv*.

**Parameters**

| Symbol | Meaning | Data source (file & column) |
|--------|---------|-----------------------------|
| \(L_p\) | Labor required per unit of product \(p\) | `Test_Dataset/Large-scale-or/Mixture_testing/Mixture2/41.csv` – column **"Labor per unit"** |
| \(M_p\) | Material required per unit of product \(p\) | same file – column **"Material per unit"** |
| \(S_p\) | Selling price per unit of product \(p\) | same file – column **"Selling Price"** |
| \(C_p\) | Variable cost per unit of product \(p\) | same file – column **"Variable Cost"** |
| \(F\) | Fixed operating cost per week | constant \(F = 4500\) |

**Variables**

- \(x_p\) – production quantity of product \(p\) (continuous, \(x_p \ge 0\)).

---

**Data Mapping Rules**

1. **Index set \(P\)**  
   \[
   P = \{\,\text{Product Name}\,\mid\, \text{row in 41.csv}\,\}
   \]

2. **Parameter extraction**  
   For each \(p \in P\):
   \[
   \begin{aligned}
   L_p &= \text{value in column "Labor per unit" for product } p,\\
   M_p &= \text{value in column "Material per unit" for product } p,\\
   S_p &= \text{value in column "Selling Price" for product } p,\\
   C_p &= \text{value in column "Variable Cost" for product } p.
   \end{aligned}
   \]

3. **Fixed cost**  
   \[
   F = 4500
   \]

These mappings ensure that every parameter in the model is directly linked to the corresponding entry in the provided CSV file. No values are enumerated in the model; the solver will read the data via the specified file path and column names.