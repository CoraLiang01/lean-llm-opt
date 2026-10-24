**Mathematical Model**

Let  

- \(I=\{i\mid i\text{ is a BookshelfID in }{\tt capacity.csv}\}\)  
- \(J=\{j\mid j\text{ is a ProductName in }{\tt products.csv}\}\)

Parameters  

- \(c_i\) – capacity of bookshelf \(i\) (from column **Capacity** of table *file_0_view_0*)  
- \(v_j\) – value of book \(j\) (from column **Value** of table *file_1_view_0*)  
- \(w_j\) – weight of book \(j\) (from column **Weight** of table *file_1_view_0*)

Decision variables  

- \(x_{ij}\in \mathbb{Z}_{\ge 0}\) – number of units of book \(j\) placed on bookshelf \(i\)

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
| \(c_i\)   | *file_0_view_0* | **Capacity** |
| \(v_j\)   | *file_1_view_0* | **Value** |
| \(w_j\)   | *file_1_view_0* | **Weight** |

All indices \(i\) and \(j\) are taken directly from the respective tables; no additional filtering or aggregation is applied.