**Symbolic Model**

Let  

* \(I\) – set of all product SKUs that start with the prefix **ZZ** (extracted from the filtered table).  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit of product \(i\) (from column **Revenue**).  
  * \(d_i\) – demand quantity of product \(i\) (from column **Demand**).  
  * \(inv_i\) – initial inventory of product \(i\) (from column **Initial Inventory**).  

Decision variable  

* \(x_i \in \mathbb{Z}_{+}\) – number of units of product \(i\) that the store plans to sell.

\[
\begin{aligned}
\max \quad & \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\quad
& 0 \le x_i \le d_i && \forall i\in I \\[2pt]
& 0 \le x_i \le inv_i && \forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{+} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping Rules**

| Symbol | Source Table | Source Column | Mapping |
|--------|--------------|---------------|---------|
| \(I\) | `file_0_view_0` | `SKU` | Each distinct SKU that satisfies the filter condition (`SKU` starts with `"ZZ"`) becomes an index element. |
| \(r_i\) | `file_0_view_0` | `Revenue` | For each SKU \(i\), \(r_i\) is the numeric value in the **Revenue** column. |
| \(d_i\) | `file_0_view_0` | `Demand` | For each SKU \(i\), \(d_i\) is the numeric value in the **Demand** column. |
| \(inv_i\) | `file_0_view_0` | `Initial Inventory` | For each SKU \(i\), \(inv_i\) is the numeric value in the **Initial Inventory** column. |

*The filter applied to the source table is: `SKU` column starts with the prefix `"ZZ"`. No other rows are included in the index set \(I\).*