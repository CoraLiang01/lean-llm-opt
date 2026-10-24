**Mathematical Model**

Let  

* \(I=\{1,2\}\) be the set of products that contain the string “27in” in the *Product Name* column.  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from the *Revenue* column)  
  * \(d_i\) – demand (from the *Demand* column)  
  * \(s_i\) – initial inventory (from the *Initial Inventory* column)  

Decision variable  

\[
x_i \;\ge\; 0 \qquad \forall i\in I
\]

*\(x_i\)* = number of units of product \(i\) that will be sold.

**Objective**

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

**Constraints**

\[
\begin{aligned}
x_i &\;\le\; d_i &&\forall i\in I \quad &&\text{(cannot sell more than demand)}\\[4pt]
x_i &\;\le\; s_i &&\forall i\in I \quad &&\text{(cannot sell more than inventory)}\\[4pt]
x_i &\;\ge\; 0 &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source | Table ID | Column | Row Index | Notes |
|-----------|--------|----------|--------|-----------|-------|
| \(r_i\) | Revenue | `file_0_view_0` | `Revenue` | 1, 2 | Revenue per unit for each product |
| \(d_i\) | Demand | `file_0_view_0` | `Demand` | 1, 2 | Maximum units that can be sold |
| \(s_i\) | Initial Inventory | `file_0_view_0` | `Initial Inventory` | 1, 2 | Available stock |
| \(I\) | Index set | – | – | – | \(I=\{1,2\}\) corresponding to the two rows returned by the filter |

*The row indices 1 and 2 refer to the two records that satisfy the filter condition “Product Name contains ‘27in’”. No other rows are included in the model.*