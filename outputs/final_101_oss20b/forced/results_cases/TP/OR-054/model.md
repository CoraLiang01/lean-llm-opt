**Mathematical Model**

Let  

* \(I=\{\,i\mid i\text{ is a ShelfID in }{\tt capacity.csv}\,\}\)  
* \(J=\{\,j\mid j\text{ is a ProductName in }{\tt products.csv}\,\}\)

Parameters  

* \(\text{Cap}_i\) – capacity of shelf \(i\) (units of weight)  
* \(\text{Val}_j\) – value per unit of product \(j\)  
* \(\text{Wgt}_j\) – weight per unit of product \(j\)

Decision variables  

\[
x_{ij}\;\ge 0 \qquad\forall\,i\in I,\;j\in J
\]

---

\[
\begin{aligned}
\max\;&\sum_{i\in I}\sum_{j\in J}\text{Val}_j\,x_{ij} \\[4pt]
\text{s.t. }&\sum_{j\in J}\text{Wgt}_j\,x_{ij}\;\le\;\text{Cap}_i
\qquad\forall\,i\in I \\[4pt]
&x_{ij}\;\ge\;0 \qquad\forall\,i\in I,\;j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(\text{Cap}_i\) | `file_0_view_0` | `Capacity` |
| \(\text{Val}_j\) | `file_1_view_0` | `Value` |
| \(\text{Wgt}_j\) | `file_1_view_0` | `Weight` |

*Index sets*  
- \(I\) is the set of all `ShelfID` values in `file_0_view_0`.  
- \(J\) is the set of all `ProductName` values in `file_1_view_0`.  

All parameters are taken directly from the corresponding columns; no additional
transformation or aggregation is performed.