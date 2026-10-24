**Mathematical Model**

Let  

* \(I\) – set of shelves (indexed by `ShelfID` from *capacity.csv*)  
* \(J\) – set of products (indexed by `ProductName` from *products.csv*)

Parameters  

\[
\begin{aligned}
\text{Capacity}_i &\;=\; \text{Capacity of shelf } i \quad (i\in I)\\
\text{Value}_j &\;=\; \text{Value of product } j \quad (j\in J)\\
\text{Weight}_j &\;=\; \text{Weight of product } j \quad (j\in J)
\end{aligned}
\]

Decision variables  

\[
x_{ij}\;=\;\text{number of units of product } j \text{ placed on shelf } i,
\qquad x_{ij}\in\mathbb{Z}_{\ge 0}\;\;(i\in I,\;j\in J)
\]

---

**Objective**

\[
\max \;\sum_{i\in I}\;\sum_{j\in J}\;\text{Value}_j\,x_{ij}
\]

---

**Constraints**

1. **Shelf capacity (weight) constraints**

\[
\sum_{j\in J}\;\text{Weight}_j\,x_{ij}\;\le\;\text{Capacity}_i,
\qquad \forall\, i\in I
\]

2. **Minimum quantity of the first product (Smartphone)**  

Let \(j^{*}\) denote the first product in *products.csv* (row 0, `ProductName` = "Smartphone").

\[
\sum_{i\in I}\;x_{i\,j^{*}}\;\ge\;5
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(I\) (ShelfID) | `file_0_view_0` | `ShelfID` |
| \(\text{Capacity}_i\) | `file_0_view_0` | `Capacity` |
| \(J\) (ProductName) | `file_1_view_0` | `ProductName` |
| \(\text{Value}_j\) | `file_1_view_0` | `Value` |
| \(\text{Weight}_j\) | `file_1_view_0` | `Weight` |

All parameters are taken directly from the corresponding CSV records; no values are enumerated in the model. The variable domain and constraints follow exactly the specifications in the query.