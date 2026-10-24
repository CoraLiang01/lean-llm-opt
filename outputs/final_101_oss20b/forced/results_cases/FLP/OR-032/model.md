**Mathematical Model**

Let  

* \(I\) – set of all products whose *Product_Name* starts with “Books”.  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**).  
  * \(d_i\) – demand quantity (from column **Demand**).  
  * \(s_i\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

\[
x_i \;\ge 0 \qquad \forall i\in I
\]

(continuous number of units of product \(i\) to be sold).

---

**Objective**

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

---

**Constraints**

\[
\begin{aligned}
&0 \;\le\; x_i \;\le\; d_i &&\forall i\in I \quad &&\text{(cannot exceed demand)}\\[4pt]
&0 \;\le\; x_i \;\le\; s_i &&\forall i\in I \quad &&\text{(cannot exceed inventory)}
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(r_i\) | Revenue | `file_0_view_0` | `Revenue` |
| \(d_i\) | Demand | `file_0_view_0` | `Demand` |
| \(s_i\) | Initial Inventory | `file_0_view_0` | `Initial Inventory` |
| Index \(i\) | Product_Name | `file_0_view_0` | `Product_Name` (filtered by prefix “Books”) |

The filter condition applied to the source table is:

```
Product_Name starts with "Books"
```

No other data or constraints are introduced beyond those explicitly required by the query.