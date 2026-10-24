**Mathematical Model**

Let  

- \(I\) : set of all products in the file.  
- For each \(i\in I\)  
  - \(r_i\) : revenue per unit (column **Revenue**)  
  - \(d_i\) : total demand (column **Demand**)  
  - \(s_i\) : initial inventory (column **Initial Inventory**)  

Decision variable  

- \(x_i \in \mathbb{Z}_{\ge 0}\) : quantity of product \(i\) the bakery plans to sell.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& x_i \le d_i && \forall i\in I \\[2pt]
& x_i \le s_i && \forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(r_i\)   | `file_0_view_0` | `Revenue` |
| \(d_i\)   | `file_0_view_0` | `Demand` |
| \(s_i\)   | `file_0_view_0` | `Initial Inventory` |

The index set \(I\) is derived from the distinct values in the **Product Name** column of the same table. No additional constraints or capacities are introduced beyond those specified.