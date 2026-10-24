**Mathematical Model**

Let  

- \(I\) : set of all products (indexed by the product name).  

Parameters (obtained from *file_0_view_0*)  

- \(r_i\) : revenue per unit of product \(i\).  
- \(d_i\) : demand quantity of product \(i\).  
- \(s_i\) : initial inventory of product \(i\).  

Decision variable  

- \(x_i \ge 0\) : number of units of product \(i\) to sell.  

\[
\begin{aligned}
\max \quad & \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\quad & 0 \;\le\; x_i \;\le\; \min\{d_i,\; s_i\} \qquad \forall i\in I .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column in Table |
|-----------|--------------|-----------------|
| \(r_i\)   | file_0_view_0 | Revenue |
| \(d_i\)   | file_0_view_0 | Demand |
| \(s_i\)   | file_0_view_0 | Initial Inventory |

The index set \(I\) is the set of distinct values in the *Product Name* column of *file_0_view_0*. No other data are required.