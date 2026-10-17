**Mathematical Model**

Let  

* \(I\) – set of Aalop products (here \(I=\{i\}\)).  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**).  
  * \(d_i\) – known demand (from column **Demand**).  
  * \(s_i\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

\[
x_i \;\ge\; 0 \qquad (\text{units of product } i \text{ to be sold})
\]

Objective  

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

Constraints  

\[
\begin{aligned}
x_i &\le d_i &&\forall i\in I &&\text{(cannot sell more than demand)}\\
x_i &\le s_i &&\forall i\in I &&\text{(cannot sell more than available inventory)}\\
x_i &\ge 0 &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Table ID          | Column in Model | Source Column (file_0_view_0) |
|-------------------|-----------------|------------------------------|
| file_0_view_0     | \(r_i\)         | Revenue                      |
| file_0_view_0     | \(d_i\)         | Demand                       |
| file_0_view_0     | \(s_i\)         | Initial Inventory            |

All parameters are taken directly from the single returned row of the filtered view. No additional data or aggregation is introduced.