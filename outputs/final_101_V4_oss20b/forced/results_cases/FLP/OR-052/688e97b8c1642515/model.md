**Mathematical Model**

Let  

- \(I=\{i\}\) be the set of bookshelves (BookshelfID).  
- \(J=\{j\}\) be the set of books (ProductName).  

Parameters  

- \(C_i\) – capacity of bookshelf \(i\) (from *capacity.csv*, column **Capacity**).  
- \(V_j\) – value of book \(j\) (from *products.csv*, column **Value**).  
- \(W_j\) – weight of book \(j\) (from *products.csv*, column **Weight**).  

Decision variables  

- \(x_{ij}\in\mathbb{Z}_{\ge 0}\) – number of units of book \(j\) placed on bookshelf \(i\).  

**Objective**

\[
\max \;\sum_{i\in I}\sum_{j\in J} V_j\,x_{ij}
\]

**Constraints**

\[
\sum_{j\in J} W_j\,x_{ij}\;\le\; C_i ,\qquad \forall i\in I
\]

\[
x_{ij}\in\mathbb{Z}_{\ge 0},\qquad \forall i\in I,\; j\in J
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(C_i\)   | `file_0_view_0` | `Capacity` |
| \(V_j\)   | `file_1_view_0` | `Value` |
| \(W_j\)   | `file_1_view_0` | `Weight` |

The index sets \(I\) and \(J\) are taken directly from the unique identifiers in the respective tables (`BookshelfID` and `ProductName`). No additional constraints or bounds are imposed beyond those stated.