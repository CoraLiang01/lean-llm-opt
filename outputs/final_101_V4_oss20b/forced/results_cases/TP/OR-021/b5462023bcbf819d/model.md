**Mathematical Model**

Let  

* \(I\) – set of products (identified by *Product Name*).  
* \(x_i\) – number of units of product \(i\) that will be fulfilled.  

Parameters (obtained from the CSV view `file_0_view_0`):

* \(\text{rev}_i\) – revenue per unit of product \(i\) (column **Revenue**).  
* \(\text{dem}_i\) – demand for product \(i\) (column **Demand**).  
* \(\text{inv}_i\) – initial inventory of product \(i\) (column **Initial Inventory**).

\[
\begin{aligned}
\max\;& \sum_{i\in I} \text{rev}_i\,x_i \\[4pt]
\text{s.t.}\;& 0 \le x_i \le \text{dem}_i && \forall i\in I \\[2pt]
& 0 \le x_i \le \text{inv}_i && \forall i\in I \\[2pt]
& x_i \;\text{continuous} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Source Column | Notes |
|-----------|--------------|---------------|-------|
| \(\text{rev}_i\) | `file_0_view_0` | `Revenue` | Float |
| \(\text{dem}_i\) | `file_0_view_0` | `Demand` | Integer |
| \(\text{inv}_i\) | `file_0_view_0` | `Initial Inventory` | Integer |
| \(I\) | `file_0_view_0` | `Product Name` | Identifier set |

All parameters are bound directly to the corresponding CSV columns; no additional aggregation or transformation is performed. The model uses the full set of products returned in the view.