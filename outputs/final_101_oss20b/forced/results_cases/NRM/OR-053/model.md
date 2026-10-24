**Mathematical Model**

Let  

- \(I\) be the set of shelves (indexed by `ShelfID`).  
- \(J\) be the set of products (indexed by `ProductName`).  

Parameters  

\[
\begin{aligned}
&\text{Capacity}[i] &&\text{(capacity of shelf }i\text{)}\\
&\text{Value}[j] &&\text{(value per unit of product }j\text{)}\\
&\text{Weight}[j] &&\text{(weight per unit of product }j\text{)}
\end{aligned}
\]

Decision variables  

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; j\in J
\]

Objective  

\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{Value}[j]\;x_{ij}
\]

Subject to  

\[
\sum_{j\in J}\text{Weight}[j]\;x_{ij}\;\le\;\text{Capacity}[i]
\qquad \forall i\in I
\]

\[
x_{ij}\;\ge\;0,\; x_{ij}\in\mathbb{Z}\qquad \forall i\in I,\; j\in J
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| `Capacity[i]` | `file_0_view_0` | `Capacity` |
| `Value[j]` | `file_1_view_0` | `Value` |
| `Weight[j]` | `file_1_view_0` | `Weight` |
| Index `i` (`ShelfID`) | `file_0_view_0` | `ShelfID` |
| Index `j` (`ProductName`) | `file_1_view_0` | `ProductName` |

All parameters are taken directly from the corresponding columns of the specified tables; no additional data or transformations are introduced.