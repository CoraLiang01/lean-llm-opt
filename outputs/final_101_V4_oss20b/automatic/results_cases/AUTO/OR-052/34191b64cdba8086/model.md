**Mathematical Model**

Let  

- \(I\) be the set of bookshelves indexed by **BookshelfID** (from *file_0_view_0*).  
- \(J\) be the set of books indexed by **ProductName** (from *file_1_view_0*).  

Parameters  

- \(C_i\) – capacity of bookshelf \(i\) (units of weight).  
- \(v_j\) – value of book \(j\).  
- \(w_j\) – weight of book \(j\).  

Decision variables  

- \(x_{ij}\) – number of units of book \(j\) placed on bookshelf \(i\).  

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{j\in J} w_j\,x_{ij} \;\le\; C_i , && \forall i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0}, && \forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Table ID | Column |
|-----------|----------|--------|
| \(C_i\) | `file_0_view_0` | `Capacity` |
| \(v_j\) | `file_1_view_0` | `Value` |
| \(w_j\) | `file_1_view_0` | `Weight` |

The index sets \(I\) and \(J\) are taken directly from the primary key columns **BookshelfID** and **ProductName** of the respective tables. No additional data or constraints are introduced beyond those specified.