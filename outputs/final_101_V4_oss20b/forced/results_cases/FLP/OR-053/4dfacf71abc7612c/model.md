**Mathematical Model**

Let  

* \(S\) – set of shelves (indexed by the values in column **ShelfID** of *capacity.csv*).  
* \(P\) – set of products (indexed by the values in column **ProductName** of *products.csv*).  

Parameters  

* \(c_s\) – capacity of shelf \(s\) (from column **Capacity**).  
* \(v_p\) – value of one unit of product \(p\) (from column **Value**).  
* \(w_p\) – weight of one unit of product \(p\) (from column **Weight**).  

Decision variables  

* \(x_{sp}\in \mathbb{Z}_{\ge 0}\) – number of units of product \(p\) placed on shelf \(s\).

\[
\begin{aligned}
\max\;& \sum_{s\in S}\sum_{p\in P} v_p\,x_{sp} \\[4pt]
\text{s.t.}\;& \sum_{p\in P} w_p\,x_{sp}\;\le\; c_s , &&\forall s\in S \\[4pt]
& x_{sp}\;\in\;\mathbb{Z}_{\ge 0}, &&\forall s\in S,\;p\in P
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Columns |
|-----------|--------------|----------------|
| \(c_s\)   | `file_0_view_0` | `ShelfID`, `Capacity` |
| \(v_p\)   | `file_1_view_0` | `ProductName`, `Value` |
| \(w_p\)   | `file_1_view_0` | `ProductName`, `Weight` |

The index sets \(S\) and \(P\) are taken from the distinct values of **ShelfID** and **ProductName** in the respective tables. No additional constraints or bounds are imposed beyond those stated.