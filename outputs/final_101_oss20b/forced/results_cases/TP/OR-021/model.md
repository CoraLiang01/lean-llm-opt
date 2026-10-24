**Mathematical model**

Let  

* \(I\) – set of all products (identified by their *Product Name*).  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**).  
  * \(d_i\) – demand quantity (from column **Demand**).  
  * \(s_i\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

\[
x_i \;\ge\; 0 \qquad \forall i\in I
\]

*\(x_i\)* = number of units of product \(i\) that will be sold.

**Objective**

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

**Constraints**

\[
\begin{aligned}
&x_i \;\le\; d_i &&\forall i\in I &&\text{(cannot sell more than demand)}\\[4pt]
&x_i \;\le\; s_i &&\forall i\in I &&\text{(cannot sell more than available inventory)}\\[4pt]
&x_i \;\ge\; 0 &&\forall i\in I
\end{aligned}
\]

---

**Data‑mapping rules**

| Parameter | Source table | Source column | Mapping description |
|-----------|--------------|---------------|----------------------|
| \(I\) | `file_0_view_0` | `Product Name` | Each distinct product name becomes an index \(i\). |
| \(r_i\) | `file_0_view_0` | `Revenue` | Revenue per unit for product \(i\). |
| \(d_i\) | `file_0_view_0` | `Demand` | Demand quantity for product \(i\). |
| \(s_i\) | `file_0_view_0` | `Initial Inventory` | Initial inventory for product \(i\). |

These mappings provide the complete set of identifiers and coefficients required for the optimization model. No additional data or constraints are introduced beyond what is specified in the query.