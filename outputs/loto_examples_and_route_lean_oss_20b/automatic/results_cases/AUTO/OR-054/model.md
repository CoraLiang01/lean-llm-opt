## Mathematical Model

**Sets**  
\(I=\{1,\dots ,10\}\) (shelves)  
\(J=\{1,\dots ,20\}\) (products)

**Parameters**  
\(c_i\) – capacity of shelf \(i\) (from *capacity.csv*)  
\(w_j\) – weight per unit of product \(j\) (from *products.csv*)  
\(v_j\) – value per unit of product \(j\) (from *products.csv*)

**Decision Variables**  
\(x_{ij}\in \mathbb{Z}_{+}\) – number of units of product \(j\) placed on shelf \(i\)

**Objective**  
\[
\max \; Z = \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij}
\]

**Constraints**  

Capacity on each shelf  
\[
\sum_{j\in J} w_j\,x_{ij}\;\le\; c_i \qquad \forall\, i\in I
\]

Non‑negativity and integrality  
\[
x_{ij}\;\ge\;0,\; x_{ij}\in \mathbb{Z}\qquad \forall\, i\in I,\; j\in J
\]

---

## Data Mapping

| Parameter | Source Table | Column | Mapping Rule |
|-----------|--------------|--------|--------------|
| \(c_i\) | `file_0_view_0` | `Capacity` | \(c_i =\) value of `Capacity` where `ShelfID = i\) |
| \(w_j\) | `file_1_view_0` | `Weight` | \(w_j =\) value of `Weight` where `ProductName = j\) |
| \(v_j\) | `file_1_view_0` | `Value` | \(v_j =\) value of `Value` where `ProductName = j\) |

Indices \(I\) and \(J\) are taken directly from the distinct values of `ShelfID` and `ProductName` in the respective tables. No additional filtering or aggregation is applied.