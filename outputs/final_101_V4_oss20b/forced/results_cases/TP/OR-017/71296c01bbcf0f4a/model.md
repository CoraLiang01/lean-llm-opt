**Mathematical Model**

Let  

* \(I\) – index set of all products whose SKU starts with “ZZ” (rows returned by the filter).  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**).  
  * \(d_i\) – demand (from column **Demand**).  
  * \(s_i\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

\[
x_i \;\ge\; 0 \qquad \forall i\in I
\]

(continuous units of product \(i\) to be sold).

**Objective**

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

**Constraints**

\[
\begin{aligned}
&x_i \;\le\; d_i &&\forall i\in I &&\text{(cannot sell more than demand)}\\
&x_i \;\le\; s_i &&\forall i\in I &&\text{(cannot sell more than available inventory)}
\end{aligned}
\]

---

**Data Mapping Rules**

| Index / Parameter | Source | Table ID | Column |
|-------------------|--------|----------|--------|
| \(I\) | SKU values | `file_0_view_0` | `SKU` |
| \(r_i\) | Revenue per unit | `file_0_view_0` | `Revenue` |
| \(d_i\) | Demand quantity | `file_0_view_0` | `Demand` |
| \(s_i\) | Initial inventory | `file_0_view_0` | `Initial Inventory` |

All mappings use the exact column names as they appear in the source file. No additional data or aggregation is introduced.