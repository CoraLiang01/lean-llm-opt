**Mathematical Model**

Let  

- \(I\) be the set of shelves (indexed by the values in column **ShelfID** of *capacity.csv*).  
- \(J\) be the set of products (indexed by the values in column **ProductName** of *products.csv*).  

Parameters  

- \(c_i\) – capacity of shelf \(i\) (column **Capacity**).  
- \(v_j\) – value per unit of product \(j\) (column **Value**).  
- \(w_j\) – weight per unit of product \(j\) (column **Weight**).  

Decision variables  

- \(x_{ij}\in \mathbb{Z}_{\ge 0}\) – number of units of product \(j\) placed on shelf \(i\).

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij} \\[4pt]
\text{s.t.}\;& \sum_{j\in J} w_j\,x_{ij}\;\le\; c_i , \qquad \forall i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0}, \qquad \forall i\in I,\; j\in J .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(c_i\) | `file_0_view_0` | `Capacity` |
| \(v_j\) | `file_1_view_0` | `Value` |
| \(w_j\) | `file_1_view_0` | `Weight` |

The index sets \(I\) and \(J\) are obtained from the distinct values of **ShelfID** and **ProductName** in the respective tables. No additional data or constraints are introduced beyond those specified in the query.