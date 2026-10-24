**Mathematical Model**

Let  

* \(I\) – index set of all products whose *Product Name* starts with “Baby” (rows returned by the query).  
* \(x_i\) – number of units of product \(i\) that the retailer plans to sell.  

Parameters (all taken from the table **file_0_view_0**):

* \(\text{Revenue}_i\) – revenue per unit of product \(i\) (column *Revenue*).  
* \(\text{Demand}_i\) – deterministic demand for product \(i\) (column *Demand*).  
* \(\text{Inv}_i\) – initial inventory of product \(i\) (column *Initial Inventory*).  

\[
\begin{aligned}
\max\;& \sum_{i\in I} \text{Revenue}_i \; x_i \\[4pt]
\text{s.t.}\;& x_i \;\le\; \text{Inv}_i , &&\forall i\in I \\[2pt]
& x_i \;\le\; \text{Demand}_i , &&\forall i\in I \\[2pt]
& x_i \;\in\; \mathbb{Z}_{\ge 0}, &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| \(\text{Revenue}_i\) | file_0_view_0 | Revenue |
| \(\text{Demand}_i\) | file_0_view_0 | Demand |
| \(\text{Inv}_i\) | file_0_view_0 | Initial Inventory |
| Index set \(I\) | file_0_view_0 | (row identifiers) |

The variable domain is specified as non‑negative integers. No other constraints are required.