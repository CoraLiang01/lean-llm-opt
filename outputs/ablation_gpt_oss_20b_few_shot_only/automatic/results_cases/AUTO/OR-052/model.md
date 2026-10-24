**Mathematical Model**

Let  

* \(I\) be the set of bookshelves (indexed by the values in *capacity.csv*).  
* \(J\) be the set of books (indexed by the values in *products.csv*).  

Parameters  

* \(c_i\) – capacity of bookshelf \(i\) (kg).  
* \(v_j\) – value of book \(j\).  
* \(w_j\) – weight of book \(j\) (kg).  

Decision variables  

* \(x_{ij}\in \mathbb{Z}_{\ge 0}\) – number of units of book \(j\) placed on bookshelf \(i\).

---

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij} \\[4pt]
\text{s.t.}\;& \sum_{j\in J} w_j\,x_{ij}\;\le\; c_i , && \forall i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0}, && \forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source File | Column Name |
|-----------|-------------|-------------|
| \(c_i\) | `Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/capacity.csv` | `Capacity` |
| \(v_j\) | `Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/products.csv` | `Value` |
| \(w_j\) | `Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/products.csv` | `Weight` |

*Index sets*  
- \(I\) is the set of all `BookshelfID` values in *capacity.csv*.  
- \(J\) is the set of all `ProductName` values in *products.csv*.  

No additional constraints or parameters are introduced beyond those explicitly required by the query.