**Symbolic Model**

Let  

- \(I\) be the set of Aalop products (here \(I=\{\text{Aalopuri}\}\)).  
- For each \(i\in I\)  
  - \(r_i\) : revenue per unit (from column **Revenue**)  
  - \(d_i\) : total demand over the horizon (from column **Demand**)  
  - \(s_i\) : initial inventory (from column **Initial Inventory**)  

Decision variable  

- \(x_i\in \mathbb{Z}_{\ge 0}\) : number of units of product \(i\) to sell.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & x_i \;\le\; d_i &&\forall i\in I \\[2pt]
             & x_i \;\le\; s_i &&\forall i\in I \\[2pt]
             & x_i \;\in\; \mathbb{Z}_{\ge 0} &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(r_i\)   | `file_0_view_0` | `Revenue` |
| \(d_i\)   | `file_0_view_0` | `Demand` |
| \(s_i\)   | `file_0_view_0` | `Initial Inventory` |

The table `file_0_view_0` is the filtered view of `RestaurantSalesreport.csv` containing only rows whose **Product Name** starts with “Aalop”. No additional data or constraints are introduced beyond those specified in the query.