## Mathematical Model

Let  

* \(I\) be the set of products (rows of the table).  
* For each \(i\in I\)  
  * \(R_i\) – revenue per unit (column **Revenue**)  
  * \(D_i\) – demand quantity (column **Demand**)  
  * \(S_i\) – initial inventory (column **Initial Inventory**)  

Decision variable  

* \(x_i\in\mathbb{Z}_{\ge 0}\) – number of units of product \(i\) that will be sold.

**Objective**

\[
\max \;\sum_{i\in I} R_i\,x_i
\]

**Constraints**

\[
\begin{aligned}
x_i &\le D_i &&\forall i\in I \quad\text{(cannot sell more than demand)}\\
x_i &\le S_i &&\forall i\in I \quad\text{(cannot sell more than available inventory)}\\
x_i &\ge 0 &&\forall i\in I
\end{aligned}
\]

---

## Data Mapping

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(R_i\)   | `file_0_view_0` | `Revenue` |
| \(D_i\)   | `file_0_view_0` | `Demand` |
| \(S_i\)   | `file_0_view_0` | `Initial Inventory` |

The index set \(I\) consists of all rows returned in `file_0_view_0` (24 products). No additional filtering or aggregation is applied.