**Mathematical Model**

Let  

- \(I=\{i\}\) be the set of bookshelves (BookshelfID).  
- \(J=\{j\}\) be the set of books (ProductName).  

Parameters  

- \(C_i\) – capacity of bookshelf \(i\).  
- \(v_j\) – value of book \(j\).  
- \(w_j\) – weight of book \(j\).  

Decision variables  

- \(x_{ij}\in \mathbb{Z}_{\ge 0}\) – number of units of book \(j\) placed on bookshelf \(i\).

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{j\in J} w_j\,x_{ij}\;\le\; C_i , && \forall i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0}, && \forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(I\) | `file_0_view_0` | `BookshelfID` |
| \(C_i\) | `file_0_view_0` | `Capacity` |
| \(J\) | `file_1_view_0` | `ProductName` |
| \(v_j\) | `file_1_view_0` | `Value` |
| \(w_j\) | `file_1_view_0` | `Weight` |

All parameters are taken directly from the corresponding CSV files; no additional limits or constants are introduced.