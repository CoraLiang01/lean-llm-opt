**Model**

Let  

- \(I=\{i\mid\text{BookshelfID}=i\}\)  (index set of bookshelves)  
- \(J=\{j\mid\text{ProductName}=j\}\)  (index set of books)

Parameters (from the source tables)

- \(c_i\) – capacity of bookshelf \(i\) (from *capacity.csv*, column **Capacity**)  
- \(v_j\) – value of book \(j\) (from *products.csv*, column **Value**)  
- \(w_j\) – weight of book \(j\) (from *products.csv*, column **Weight**)

Decision variables

- \(x_{ij}\in\mathbb{Z}_{\ge 0}\) – number of units of book \(j\) placed on bookshelf \(i\)

**Objective**

\[
\max \;\sum_{i\in I}\sum_{j\in J} v_j\,x_{ij}
\]

**Constraints**

1. Capacity limits for each bookshelf  
\[
\sum_{j\in J} w_j\,x_{ij}\;\le\; c_i \qquad \forall i\in I
\]

2. Non‑negativity and integrality  
\[
x_{ij}\in\mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; j\in J
\]

---

**Data Mapping**

| Index | Source Table | Column |
|-------|--------------|--------|
| \(i\) | *capacity.csv* | **BookshelfID** |
| \(c_i\) | *capacity.csv* | **Capacity** |
| \(j\) | *products.csv* | **ProductName** |
| \(v_j\) | *products.csv* | **Value** |
| \(w_j\) | *products.csv* | **Weight** |

All parameters are taken directly from the corresponding columns of the specified source tables; no additional data or transformations are introduced.