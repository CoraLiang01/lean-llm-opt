**Mathematical Model**

Let  

- \(I\) be the set of products whose *Product Name* starts with “Baby”.  
- For each \(i\in I\)  
  - \(R_i\) = revenue per unit of product \(i\)  
  - \(D_i\) = demand for product \(i\)  
  - \(I_i\) = initial inventory of product \(i\)  

Decision variable  

- \(x_i \ge 0\) = number of units of product \(i\) to sell (continuous).

\[
\begin{aligned}
\max\;& \sum_{i\in I} R_i\,x_i \\[4pt]
\text{s.t. } & x_i \;\le\; D_i &&\forall i\in I \\[2pt]
             & x_i \;\le\; I_i &&\forall i\in I \\[2pt]
             & x_i \;\ge\; 0 &&\forall i\in I
\end{aligned}
\]

---

**Data‑Mapping Rules**

| Symbol | Source Table | Source Column | Filter Applied |
|--------|--------------|---------------|----------------|
| \(I\) | `file_0_view_0` | row index | `Product Name` starts with `"Baby"` |
| \(R_i\) | `file_0_view_0` | `Revenue` | same filter |
| \(D_i\) | `file_0_view_0` | `Demand` | same filter |
| \(I_i\) | `file_0_view_0` | `Initial Inventory` | same filter |
| \(x_i\) | decision variable | – | – |

*All parameters are taken directly from the filtered view `file_0_view_0` of `Salesdata.csv`.*