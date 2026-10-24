**Mathematical Model**

Let  

- \(I\) be the set of products whose *Product Name* starts with **S700_**.  
- For each \(i\in I\)  
  - \(r_i\) = revenue per unit (column **Revenue**)  
  - \(d_i\) = demand quantity (column **Demand**)  
  - \(v_i\) = initial inventory (column **Initial Inventory**)  

Decision variable  

- \(x_i \ge 0\) = units of product \(i\) to be sold.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le d_i &&\forall i\in I \\[2pt]
             & 0 \le x_i \le v_i &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source | Table ID | Column | Filter |
|--------|--------|----------|--------|--------|
| \(I\) | Set of product names | `file_0_view_0` | `Product Name` | `Product Name` starts with `"S700_"` |
| \(r_i\) | Revenue per unit | `file_0_view_0` | `Revenue` | same row as \(i\) |
| \(d_i\) | Demand quantity | `file_0_view_0` | `Demand` | same row as \(i\) |
| \(v_i\) | Initial inventory | `file_0_view_0` | `Initial Inventory` | same row as \(i\) |

All parameters are taken directly from the filtered rows of the CSV file; no numerical values are enumerated in the model.