**Mathematical Model**

Sets  
\[
I \;=\; \{\text{ShelfID}\}\quad\text{(from capacity.csv)}\\
J \;=\; \{\text{ProductName}\}\quad\text{(from products.csv)}
\]

Parameters  
\[
\begin{aligned}
\text{Capacity}[i] &\;=\; \text{Capacity of shelf }i \quad
  \text{(file_0_view_0, column Capacity)}\\
\text{Value}[j] &\;=\; \text{value of product }j \quad
  \text{(file_1_view_0, column Value)}\\
\text{Weight}[j] &\;=\; \text{weight of product }j \quad
  \text{(file_1_view_0, column Weight)}\\
\text{FirstProduct} &\;=\; \text{the first product in J (row 0 of products.csv)}\\
\end{aligned}
\]

Decision Variables  
\[
x_{ij}\;\;=\;\;\text{number of units of product }j\text{ placed on shelf }i,
\qquad x_{ij}\in\mathbb{Z}_{+}\;\;\forall i\in I,\;j\in J
\]

Objective  
\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{Value}[j]\;x_{ij}
\]

Constraints  

1. **Capacity constraints**  
\[
\sum_{j\in J}\text{Weight}[j]\;x_{ij}\;\le\;\text{Capacity}[i]
\quad\forall i\in I
\]

2. **Minimum quantity of first product**  
\[
\sum_{i\in I}x_{i,\text{FirstProduct}}\;\ge\;5
\]

---

**Data Mapping**

| Parameter | Source Table | Column | Notes |
|-----------|--------------|--------|-------|
| `Capacity[i]` | `file_0_view_0` | `Capacity` | row identified by `ShelfID` |
| `Value[j]` | `file_1_view_0` | `Value` | row identified by `ProductName` |
| `Weight[j]` | `file_1_view_0` | `Weight` | row identified by `ProductName` |
| `FirstProduct` | `file_1_view_0` | `ProductName` | row 0 (first record) |

All other data are derived from the index sets `I` and `J` as listed above.