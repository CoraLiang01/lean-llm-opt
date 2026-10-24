## Mathematical Model

Let  

* \(I\) – set of tablet models that satisfy the filter  
  (`Product Name` contains “TABLET”).  
  \(I=\{i\mid\) row \(i\) is in the 27 returned rows\(\}\).

Parameters (all taken from table **file_0_view_0**):

* \(r_i\) – revenue per unit of model \(i\)  (column **Revenue**).  
* \(d_i\) – demand quantity of model \(i\)  (column **Demand**).  
* \(s_i\) – initial inventory of model \(i\)  (column **Initial Inventory**).

Decision variables:

* \(x_i\) – number of units of model \(i\) to sell.  
  \(x_i\in\mathbb{Z}_{+}\).

**Objective**

\[
\max \sum_{i\in I} r_i\,x_i
\]

**Constraints**

\[
\begin{aligned}
&0 \le x_i \le d_i &&\forall i\in I \quad\text{(cannot exceed demand)}\\
&0 \le x_i \le s_i &&\forall i\in I \quad\text{(cannot exceed inventory)}\\
\end{aligned}
\]

---

## Data Mapping Rules

| Index Set | Definition |
|-----------|------------|
| \(I\) | All rows returned by the filter on **Product Name** containing “TABLET” in table **file_0_view_0**. |

| Parameter | Source Table | Column | Notes |
|-----------|--------------|--------|-------|
| \(r_i\) | **file_0_view_0** | **Revenue** | Revenue per unit for model \(i\). |
| \(d_i\) | **file_0_view_0** | **Demand** | Deterministic demand for model \(i\). |
| \(s_i\) | **file_0_view_0** | **Initial Inventory** | Available inventory for model \(i\). |

| Variable | Domain |
|----------|--------|
| \(x_i\) | Integer, \(x_i \ge 0\) |

The filter condition applied to build \(I\) is:

```
WHERE "Product Name" CONTAINS 'TABLET'
```

All other rows in the original file are excluded from the model. No additional constraints or parameters are introduced beyond those explicitly stated.