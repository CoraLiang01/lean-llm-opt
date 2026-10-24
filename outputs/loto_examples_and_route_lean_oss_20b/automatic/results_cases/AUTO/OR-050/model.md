**Mathematical Model**

Let  

* \(I=\{\,\text{ShelfID}\,\}\) – set of shelves (from *capacity.csv*).  
* \(J=\{\,\text{ProductName}\,\}\) – set of products (from *products.csv*).  

Parameters (all numeric):

* \(\text{Cap}[i]\) – capacity of shelf \(i\).  
* \(\text{Val}[j]\) – value of product \(j\).  
* \(\text{Wgt}[j]\) – weight of product \(j\).  

Decision variables:

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; j\in J
\]

Objective (maximize total value):

\[
\max \;\sum_{i\in I}\;\sum_{j\in J}\text{Val}[j]\;x_{ij}
\]

Constraints  

1. **Shelf capacity**  
\[
\sum_{j\in J}\text{Wgt}[j]\;x_{ij}\;\le\;\text{Cap}[i]\qquad \forall i\in I
\]

2. **Minimum quantity of the first product**  
(The first product in *products.csv* is “Smartphone”.)  
\[
\sum_{i\in I}x_{i,\text{Smartphone}}\;\ge\;5
\]

---

**Data Mapping**

| Parameter | Table ID | Column |
|-----------|----------|--------|
| \(\text{Cap}[i]\) | `file_0_view_0` | `Capacity` |
| \(\text{Val}[j]\) | `file_1_view_0` | `Value` |
| \(\text{Wgt}[j]\) | `file_1_view_0` | `Weight` |
| Index set \(I\) | `file_0_view_0` | `ShelfID` |
| Index set \(J\) | `file_1_view_0` | `ProductName` |

All other data (e.g., the identity of the first product) are derived from the order of rows in *products.csv*; the constraint uses the literal product name “Smartphone”.