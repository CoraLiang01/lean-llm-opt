**Mathematical Model**

Let  

* \(I\) be the set of all products (identified by *Product Name*).  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**).  
  * \(d_i\) – demand quantity (from column **Demand**).  
  * \(s_i\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

* \(x_i\) – number of units of product \(i\) to be sold (integer, \(\ge 0\)).

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& 0 \le x_i \le d_i &&\forall i\in I \\[2pt]
& 0 \le x_i \le s_i &&\forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_+ &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(r_i\) | Revenue | file_0_view_0 | Revenue |
| \(d_i\) | Demand | file_0_view_0 | Demand |
| \(s_i\) | Initial Inventory | file_0_view_0 | Initial Inventory |
| \(I\) | Product Name | file_0_view_0 | Product Name |

All indices and parameters are taken directly from the rows of the CSV file *WomenClothingEcommerceSalesData.csv* (table_id `file_0_view_0`). No numerical values are enumerated; the model references the data symbolically.