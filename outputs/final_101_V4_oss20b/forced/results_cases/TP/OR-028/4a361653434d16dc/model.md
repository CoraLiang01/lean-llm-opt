**Mathematical Model**

Let  

* \(I\) – set of products (indexed by the *Product Name* values in the table).  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**).  
  * \(d_i\) – demand quantity (from column **Demand**).  
  * \(s_i\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

* \(x_i \ge 0\) – number of units of product \(i\) to be fulfilled.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& x_i \;\le\; d_i &&\forall i\in I \\[2pt]
            & x_i \;\le\; s_i &&\forall i\in I \\[2pt]
            & x_i \;\ge\; 0 &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(r_i\)   | Revenue | file_0_view_0 | Revenue |
| \(d_i\)   | Demand | file_0_view_0 | Demand |
| \(s_i\)   | Initial Inventory | file_0_view_0 | Initial Inventory |
| \(I\)     | Product Name | file_0_view_0 | Product Name |

*Each product name in the **Product Name** column defines a distinct index \(i\).  
All numeric values are taken directly from the corresponding columns without modification.*