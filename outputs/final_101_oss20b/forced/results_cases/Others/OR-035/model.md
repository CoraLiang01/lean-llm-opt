**Mathematical Model**

Let  

* \(I=\{1,\dots ,10\}\) be the set of bread types (index by the order in *products.csv*).  
* \(x_i\in \mathbb{Z}_{\ge 0}\) – number of units of bread type \(i\) ordered each day.  

Parameters (all taken directly from the supplied tables)

* \(v_i\) – expected profit per unit of bread type \(i\) (column **Value** in *products.csv*).  
* \(w_i\) – weight (storage space) per unit of bread type \(i\) (column **Weight** in *products.csv*).  
* \(C\) – total storage capacity (column **Capacity** in *capacity.csv*).  

The bakery’s problem is

\[
\begin{aligned}
\max_{x}\quad & \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\quad & \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column(s) | Notes |
|-----------|--------------|-----------|-------|
| \(C\) | `file_0_view_0` | `Capacity` | Single value 180 |
| \(v_i\) | `file_1_view_0` | `Value` | Profit per unit for each product |
| \(w_i\) | `file_1_view_0` | `Weight` | Weight per unit for each product |
| \(I\) | `file_1_view_0` | `ProductName` | Index set of the 10 products |

All parameters are taken verbatim from the corresponding columns; no additional processing or enumeration of individual rows is performed. The model uses the full set of products returned by the query.