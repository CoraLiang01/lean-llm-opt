**Mathematical Model**

Let  

* \(I\) – set of shelves (indexed by the values in column **ShelfID** of *file_0_view_0*).  
* \(J\) – set of products (indexed by the values in column **ProductName** of *file_1_view_0*).  

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
x_{ij}\;\;=\;\;\text{number of units of product } j \text{ placed on shelf } i,
\qquad i\in I,\; j\in J
\]

\[
x_{ij}\in \mathbb{Z}_{\ge 0}
\]

---

**Objective**

\[
\max \;\; \sum_{i\in I}\;\sum_{j\in J}\; \text{Value}_j \; x_{ij}
\]

---

**Constraints**

1. **Shelf‑capacity (weight) constraints**

\[
\sum_{j\in J}\; \text{Weight}_j \; x_{ij}\;\le\; \text{Capacity}_i,
\qquad \forall\, i\in I
\]

2. **Non‑negativity (implicit in variable domain)**  

\[
x_{ij}\;\ge\;0,\qquad \forall\, i\in I,\; j\in J
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(\text{Capacity}_i\) | `file_0_view_0` | `Capacity` |
| \(\text{Value}_j\) | `file_1_view_0` | `Value` |
| \(\text{Weight}_j\) | `file_1_view_0` | `Weight` |
| Index \(i\) | `file_0_view_0` | `ShelfID` |
| Index \(j\) | `file_1_view_0` | `ProductName` |

All parameters and indices are taken directly from the corresponding tables; no additional data or assumptions are introduced.