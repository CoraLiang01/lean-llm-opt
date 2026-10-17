**Mathematical Model**

Let  

- \(I\) be the set of 27‑inch products (indexed by their *Product Name*).  
- For each \(i\in I\)  
  - \(r_i\) : revenue per unit (from column **Revenue**)  
  - \(d_i\) : demand quantity (from column **Demand**)  
  - \(v_i\) : initial inventory (from column **Initial Inventory**)  
  - \(x_i\) : number of units of product \(i\) to be sold (decision variable).

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le d_i && \forall i\in I \\[2pt]
             & 0 \le x_i \le v_i && \forall i\in I \\[2pt]
             & x_i \in \mathbb{Z}_{+} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(I\) | Product identifiers | `file_0_view_0` | `Product Name` |
| \(r_i\) | Revenue per unit | `file_0_view_0` | `Revenue` |
| \(d_i\) | Demand quantity | `file_0_view_0` | `Demand` |
| \(v_i\) | Initial inventory | `file_0_view_0` | `Initial Inventory` |

*All values are taken directly from the filtered rows of `Salesorders.csv` where `Product Name` contains “27in”. No numerical values are enumerated in the model.*