**Mathematical Model**

Let  

* \(I=\{1,\dots ,10\}\) – set of display (shelf) identifiers (from *capacity.csv*).  
* \(J=\{1,\dots ,20\}\) – set of product identifiers in the order they appear in *products.csv*  
  (product 1 = “Smartphone”, product 2 = “Laptop”, …).  

Decision variables  

\[
x_{ij}\;\ge 0\quad\text{(integer)}\qquad \forall i\in I,\; j\in J
\]

*\(x_{ij}\)* = number of units of product \(j\) placed on display \(i\).

Parameters  

\[
\begin{aligned}
\text{capacity}_i &\;=\;\text{Capacity of shelf }i \quad (i\in I)\\
\text{value}_j    &\;=\;\text{Value of product }j \quad (j\in J)\\
\text{weight}_j   &\;=\;\text{Weight of product }j \quad (j\in J)
\end{aligned}
\]

**Objective**

\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{value}_j\,x_{ij}
\]

**Constraints**

1. **Weight capacity of each display**

\[
\sum_{j\in J}\text{weight}_j\,x_{ij}\;\le\;\text{capacity}_i
\qquad \forall i\in I
\]

2. **Minimum quantity of the first product (Smartphone)**

\[
\sum_{i\in I}x_{i1}\;\ge\;5
\]

3. **Non‑negativity**

\[
x_{ij}\;\ge\;0 \quad \forall i\in I,\; j\in J
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Source Column | Mapping Expression |
|-----------|--------------|---------------|---------------------|
| `capacity_i` | `file_0_view_0` | `Capacity` | `capacity_i = Capacity where ShelfID = i` |
| `value_j` | `file_1_view_0` | `Value` | `value_j = Value where ProductName = j` |
| `weight_j` | `file_1_view_0` | `Weight` | `weight_j = Weight where ProductName = j` |

*Indices `i` and `j` are taken directly from the `ShelfID` and the order of rows in the respective CSV files.*