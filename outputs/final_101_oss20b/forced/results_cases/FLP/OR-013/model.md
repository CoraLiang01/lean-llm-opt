**Mathematical Model**

Let  

* \(I\) – set of all products whose *Product Name* contains “4U” (the 18 rows returned).  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**).  
  * \(d_i\) – demand (from column **Demand**).  
  * \(s_i\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

* \(x_i \in \mathbb{Z}_{+}\) – number of units of product \(i\) to sell.

---

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& x_i \;\le\; d_i &&\forall i\in I \\[2pt]
& x_i \;\le\; s_i &&\forall i\in I \\[2pt]
& x_i \;\in\; \mathbb{Z}_{+} &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(I\) | set of product names | `file_0_view_0` | `Product Name` |
| \(r_i\) | revenue per unit | `file_0_view_0` | `Revenue` |
| \(d_i\) | demand | `file_0_view_0` | `Demand` |
| \(s_i\) | initial inventory | `file_0_view_0` | `Initial Inventory` |
| \(x_i\) | decision variable | – | – |

All parameters are taken directly from the filtered rows of the CSV file; no numerical values are enumerated in the model.