**Mathematical Model**

Let  

* \(I\) – set of products that contain “27in” in their name.  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**).  
  * \(d_i\) – demand quantity (from column **Demand**).  
  * \(s_i\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

* \(x_i \ge 0\) – number of units of product \(i\) to be sold (continuous, can be integer if required).

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & x_i \;\le\; d_i &&\forall i\in I \quad\text{(cannot exceed demand)}\\
            & x_i \;\le\; s_i &&\forall i\in I \quad\text{(cannot exceed available inventory)}\\
            & x_i \;\ge\; 0 &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Table ID          | Column in Model | Source Column (CSV) |
|-------------------|-----------------|---------------------|
| `file_0_view_0`   | \(I\) (product name) | `Product Name` |
| `file_0_view_0`   | \(r_i\) | `Revenue` |
| `file_0_view_0`   | \(d_i\) | `Demand` |
| `file_0_view_0`   | \(s_i\) | `Initial Inventory` |

All parameters are taken directly from the filtered rows of the CSV file; no numerical values are enumerated in the model.