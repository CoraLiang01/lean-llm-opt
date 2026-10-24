**Mathematical Model**

Let  

- \(I\) be the set of bookshelves.  
- \(J\) be the set of books.  

Parameters  

- \(c_i\) – capacity of bookshelf \(i\) (from *capacity.csv*).  
- \(v_j\) – value of book \(j\) (from *products.csv*).  
- \(w_j\) – weight of book \(j\) (from *products.csv*).  

Decision variables  

- \(x_{ij}\in \mathbb{Z}_{\ge 0}\) – number of units of book \(j\) placed on bookshelf \(i\).

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{j\in J} w_j\,x_{ij} \;\le\; c_i , && \forall i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0}, && \forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(c_i\) | `file_0_view_0` | `Capacity` |
| \(v_j\) | `file_1_view_0` | `Value` |
| \(w_j\) | `file_1_view_0` | `Weight` |

*Indices*  
- \(i\) ranges over `BookshelfID` values in `file_0_view_0`.  
- \(j\) ranges over `ProductName` values in `file_1_view_0`.