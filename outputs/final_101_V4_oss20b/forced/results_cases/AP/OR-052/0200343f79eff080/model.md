**Mathematical Model**

Let  

- \(I\) be the set of bookshelves indexed by **BookshelfID** from *capacity.csv*.  
- \(J\) be the set of books indexed by **ProductName** from *products.csv*.  

Parameters  

- \(c_i\) – capacity of bookshelf \(i\) (units of weight).  
- \(v_j\) – value of one unit of book \(j\).  
- \(w_j\) – weight of one unit of book \(j\).  

Decision variables  

- \(x_{ij}\in \mathbb{Z}_{\ge 0}\) – number of units of book \(j\) placed on bookshelf \(i\).

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij} \\[4pt]
\text{s.t.}\;& \sum_{j\in J} w_j\,x_{ij}\;\le\; c_i \qquad \forall i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(c_i\)   | `file_0_view_0` | `Capacity` |
| \(v_j\)   | `file_1_view_0` | `Value` |
| \(w_j\)   | `file_1_view_0` | `Weight` |

Index sets are derived from the primary key columns of the respective tables (`BookshelfID` and `ProductName`).