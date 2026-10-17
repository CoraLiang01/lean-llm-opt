**Mathematical Model**

\[
\begin{aligned}
\max\;&\sum_{i\in I} \text{rev}_i\,x_i \\
\text{s.t. }&x_i \;\le\; \text{dem}_i &&\forall i\in I \\
&x_i \;\le\; \text{inv}_i &&\forall i\in I \\
&x_i \;\ge\; 0 &&\forall i\in I
\end{aligned}
\]

*Decision variables*  
\(x_i\) – number of units of product \(i\) that will be sold (continuous, non‑negative).

*Sets*  
\(I\) – set of products whose *Product Name* contains the string “27in” (obtained by the filter in the query).

*Parameters*  
\(\text{rev}_i\) – revenue per unit of product \(i\).  
\(\text{dem}_i\) – deterministic demand for product \(i\).  
\(\text{inv}_i\) – initial inventory of product \(i\).

---

**Data Mapping Rules**

| Parameter | Source Table | Source Columns | Mapping |
|-----------|--------------|----------------|---------|
| \(\text{rev}_i\) | `file_0_view_0` | `Revenue` | `rev_i = Revenue` |
| \(\text{dem}_i\) | `file_0_view_0` | `Demand` | `dem_i = Demand` |
| \(\text{inv}_i\) | `file_0_view_0` | `Initial Inventory` | `inv_i = Initial Inventory` |
| \(I\) | `file_0_view_0` | `Product Name` | Include row if `Product Name` contains `"27in"` |

All numeric values are taken directly from the corresponding columns of the filtered view `file_0_view_0`. No additional data or aggregation is introduced.