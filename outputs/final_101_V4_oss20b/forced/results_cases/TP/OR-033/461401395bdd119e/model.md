**Mathematical Model**

Let  

* \(I\) – set of Baby products (rows returned by the filter).  
* \(x_i\) – number of units of product \(i\) to be fulfilled, \(x_i\ge 0\).

Parameters (obtained from the filtered view `file_0_view_0`):

* \(\text{rev}_i\) – revenue per unit of product \(i\) (column **Revenue**).  
* \(\text{dem}_i\) – demand for product \(i\) (column **Demand**).  
* \(\text{inv}_i\) – initial inventory of product \(i\) (column **Initial Inventory**).

The model is

\[
\begin{aligned}
\max_{x}\;& \sum_{i\in I}\text{rev}_i\,x_i \\[4pt]
\text{s.t. }& 0 \le x_i \le \text{dem}_i && \forall i\in I \\[2pt]
& 0 \le x_i \le \text{inv}_i && \forall i\in I .
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Column Name | Notes |
|-----------|--------------|-------------|-------|
| \(\text{rev}_i\) | `file_0_view_0` | `Revenue` | Revenue per unit for product \(i\). |
| \(\text{dem}_i\) | `file_0_view_0` | `Demand` | Deterministic demand for product \(i\). |
| \(\text{inv}_i\) | `file_0_view_0` | `Initial Inventory` | Available inventory for product \(i\). |

The index set \(I\) is defined by the rows returned after applying the filter `Product Name contains "Baby"` on `EuropeSalesRecords.csv`. No additional indices or parameters are required.