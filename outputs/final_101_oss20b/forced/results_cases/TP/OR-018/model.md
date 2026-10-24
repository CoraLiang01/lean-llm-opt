**Mathematical Model**

Let  

* \(I\) – set of all products whose *Product Name* starts with “Baby”.  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**).  
  * \(d_i\) – demand (from column **Demand**).  
  * \(s_i\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

\[
x_i \;\ge\; 0 \qquad \forall i\in I
\]

*\(x_i\)* = number of units of product \(i\) that the retailer plans to sell.

**Objective**

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

**Constraints**

\[
\begin{aligned}
x_i &\;\le\; d_i &&\forall i\in I \quad &&\text{(cannot sell more than demand)}\\[4pt]
x_i &\;\le\; s_i &&\forall i\in I \quad &&\text{(cannot sell more than available inventory)}\\[4pt]
x_i &\;\ge\; 0   &&\forall i\in I
\end{aligned}
\]

---

**Data‑Mapping Rules**

| Parameter | Source Table | Source Column | Mapping |
|-----------|--------------|---------------|---------|
| \(I\) | `file_0_view_0` | `Product Name` | All rows where `Product Name` starts with “Baby” |
| \(r_i\) | `file_0_view_0` | `Revenue` | `r_i = Revenue` for row \(i\) |
| \(d_i\) | `file_0_view_0` | `Demand` | `d_i = Demand` for row \(i\) |
| \(s_i\) | `file_0_view_0` | `Initial Inventory` | `s_i = Initial Inventory` for row \(i\) |

*No additional data or constraints are introduced beyond those specified in the query.*