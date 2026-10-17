**Mathematical Model**

Let  

- \(I\) be the set of *Organ* products (rows returned by the filter).  
- For each \(i\in I\)  
  - \(R_i\) : revenue per unit (from column **Revenue**)  
  - \(D_i\) : demand quantity (from column **Demand**)  
  - \(I_i\) : initial inventory (from column **Initial Inventory**)  

Decision variable  

- \(x_i \ge 0\) : number of units of product \(i\) to be fulfilled.

\[
\begin{aligned}
\max\;& \sum_{i\in I} R_i\,x_i \\[4pt]
\text{s.t. } & x_i \;\le\; D_i && \forall i\in I \\[2pt]
             & x_i \;\le\; I_i && \forall i\in I \\[2pt]
             & x_i \;\ge\; 0 && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(I\) | Set of rows after filter | `file_0_view_0` | `Sub Category` |
| \(R_i\) | Revenue per unit | `file_0_view_0` | `Revenue` |
| \(D_i\) | Demand quantity | `file_0_view_0` | `Demand` |
| \(I_i\) | Initial inventory | `file_0_view_0` | `Initial Inventory` |

No additional parameters or constraints are required beyond those derived from the supplied data.