## Mathematical Model

**Sets**  
\(I=\{\,i\mid i\text{ is a ShelfID in }{\tt capacity.csv}\,\}\)  
\(J=\{\,j\mid j\text{ is a ProductName in }{\tt products.csv}\,\}\)  
Let \(j^{*}\) denote the first product in the list (index 1, “Smartphone”).

**Parameters**  
\(C_i\) – capacity of shelf \(i\) (kg)  (from column **Capacity** of table file_0_view_0)  
\(W_j\) – weight of one unit of product \(j\) (kg) (from column **Weight** of table file_1_view_0)  
\(V_j\) – value of one unit of product \(j\) (yuan) (from column **Value** of table file_1_view_0)

**Decision Variables**  
\(x_{ij}\in\mathbb{Z}_{+}\) – number of units of product \(j\) placed on shelf \(i\).

**Objective**  
\[
\max \;\sum_{i\in I}\sum_{j\in J} V_j\,x_{ij}
\]

**Constraints**

1. **Shelf capacity (weight)**  
\[
\sum_{j\in J} W_j\,x_{ij}\;\le\; C_i \qquad \forall\,i\in I
\]

2. **Minimum quantity of first product**  
\[
\sum_{i\in I} x_{i\,j^{*}}\;\ge\;5
\]

3. **Non‑negativity**  
\[
x_{ij}\;\ge\;0 \qquad \forall\,i\in I,\;j\in J
\]

---

## Data Mapping Rules

| Parameter | Source Table | Source Column | Mapping |
|-----------|--------------|---------------|---------|
| \(C_i\) | file_0_view_0 | Capacity | \(C_i =\) value of **Capacity** where **ShelfID** = \(i\) |
| \(W_j\) | file_1_view_0 | Weight | \(W_j =\) value of **Weight** where **ProductName** = \(j\) |
| \(V_j\) | file_1_view_0 | Value | \(V_j =\) value of **Value** where **ProductName** = \(j\) |

*Index sets \(I\) and \(J\) are the full sets of ShelfIDs and ProductNames returned by the respective tables; no additional filtering is applied.*