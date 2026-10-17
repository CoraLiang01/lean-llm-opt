**Mathematical Model**

Let  

* \(I\) – set of all products whose name starts with “Books” (obtained from the filtered view).  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (parameter).  
  * \(d_i\) – demand quantity (parameter).  
  * \(s_i\) – initial inventory (parameter).  

Decision variable  

* \(x_i \ge 0\) – number of units of product \(i\) that will be sold (fulfilled).

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & x_i \;\le\; s_i &&\forall i\in I \\[2pt]
             & x_i \;\le\; d_i &&\forall i\in I \\[2pt]
             & x_i \;\ge\; 0   &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(r_i\)   | Revenue | `file_0_view_0` | `Revenue` |
| \(d_i\)   | Demand | `file_0_view_0` | `Demand` |
| \(s_i\)   | Initial Inventory | `file_0_view_0` | `Initial Inventory` |
| Index \(i\) | Product name | `file_0_view_0` | `Product_Name` (filtered where `Product_Name` starts with “Books”) |

The index set \(I\) is the collection of all `Product_Name` values that satisfy the filter condition in the observation. No numerical values are enumerated; the model references the parameters directly from the source table.