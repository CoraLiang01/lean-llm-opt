**Symbolic Model**

Let  

- \(I\) : set of all products (rows of *RetailSalesDataset.csv*).  
- For each \(i\in I\)  
  - \(R_i\) : revenue per unit (column **Revenue**).  
  - \(D_i\) : deterministic demand (column **Demand**).  
  - \(S_i\) : initial inventory (column **Initial Inventory**).  

Decision variable  

- \(x_i \in \mathbb{Z}_{\ge 0}\) : number of units of product \(i\) to sell.

\[
\begin{aligned}
\max\;& \sum_{i\in I} R_i\,x_i \\[4pt]
\text{s.t.}\;& x_i \le D_i &&\forall i\in I \\[2pt]
& x_i \le S_i &&\forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{\ge 0} &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(R_i\)   | `file_0_view_0` | `Revenue` |
| \(D_i\)   | `file_0_view_0` | `Demand` |
| \(S_i\)   | `file_0_view_0` | `Initial Inventory` |

The index set \(I\) is the set of all rows returned from `file_0_view_0`. No additional data or constraints are introduced beyond those specified in the query.