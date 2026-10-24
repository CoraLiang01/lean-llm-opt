**Mathematical Model**

Let  

- \(I\) be the set of shelves indexed by the values of **ShelfID** in *file_0_view_0*.  
- \(J\) be the set of products indexed by the values of **ProductName** in *file_1_view_0*.  

Parameters  

\[
\begin{aligned}
\text{Capacity}_i &\quad\text{(weight capacity of shelf }i), && i\in I\\
\text{Value}_j &\quad\text{(value of product }j), && j\in J\\
\text{Weight}_j &\quad\text{(weight of product }j), && j\in J
\end{aligned}
\]

Decision variables  

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\qquad i\in I,\; j\in J
\]

---

**Objective**

\[
\max \;\sum_{i\in I}\;\sum_{j\in J}\;\text{Value}_j \; x_{ij}
\]

---

**Constraints**

1. **Shelf capacity**

\[
\sum_{j\in J}\;\text{Weight}_j \; x_{ij}\;\le\;\text{Capacity}_i
\qquad \forall\, i\in I
\]

---

**Data Mapping**

| Parameter | Table ID | Column |
|-----------|----------|--------|
| `Capacity_i` | `file_0_view_0` | `Capacity` |
| `Value_j` | `file_1_view_0` | `Value` |
| `Weight_j` | `file_1_view_0` | `Weight` |
| Index `i` (`ShelfID`) | `file_0_view_0` | `ShelfID` |
| Index `j` (`ProductName`) | `file_1_view_0` | `ProductName` |

All indices, parameters, and variables are defined exactly as they appear in the supplied tables. No additional data or constraints are introduced.