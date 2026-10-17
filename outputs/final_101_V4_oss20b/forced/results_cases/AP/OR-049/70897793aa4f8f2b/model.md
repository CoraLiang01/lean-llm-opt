**Mathematical Model**

Let  

- \(I\) be the set of shelf identifiers.  
- \(J\) be the set of product names.  

Parameters  

\[
\begin{aligned}
\text{capacity}[i] &\quad i\in I &&\text{(weight capacity of shelf }i)\\
\text{value}[j] &\quad j\in J &&\text{(value per unit of product }j)\\
\text{weight}[j] &\quad j\in J &&\text{(weight per unit of product }j)
\end{aligned}
\]

Decision variables  

\[
x_{ij}\in \mathbb{Z}_{+}\quad \forall i\in I,\; j\in J
\]

Objective  

\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{value}[j]\;x_{ij}
\]

Subject to  

\[
\sum_{j\in J}\text{weight}[j]\;x_{ij}\;\le\;\text{capacity}[i]\quad \forall i\in I
\]

\[
x_{ij}\;\ge\;0,\; x_{ij}\in \mathbb{Z}\quad \forall i\in I,\; j\in J
\]

---

**Data Mapping**

| Parameter | Source Table | Key Column | Value Column | Mapping Rule |
|-----------|--------------|------------|--------------|--------------|
| `capacity[i]` | `file_0_view_0` | `ShelfID` | `Capacity` | `capacity[i] = record where ShelfID = i` |
| `value[j]` | `file_1_view_0` | `ProductName` | `Value` | `value[j] = record where ProductName = j` |
| `weight[j]` | `file_1_view_0` | `ProductName` | `Weight` | `weight[j] = record where ProductName = j` |

Index sets are derived from the distinct values of the key columns:

- \(I = \{\,\text{ShelfID values from }file_0_view_0\,\}\)
- \(J = \{\,\text{ProductName values from }file_1_view_0\,\}\)

All parameters are taken directly from the CSV records; no numerical values are enumerated in the model.