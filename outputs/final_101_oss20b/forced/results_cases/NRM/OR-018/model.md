**Mathematical Model**

Let  

- \(I\) be the set of all products whose *Product Name* contains the string **“Baby”** (as defined by the filter in the query).  
- For each \(i\in I\)  
  - \(r_i\) – revenue per unit (from column **Revenue**).  
  - \(d_i\) – total demand over the horizon (from column **Demand**).  
  - \(I_i\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

- \(x_i\in \mathbb{Z}_{+}\) – number of units of product \(i\) to be sold.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& x_i \;\le\; d_i &&\forall i\in I \\[2pt]
& x_i \;\le\; I_i &&\forall i\in I \\[2pt]
& x_i \;\ge\; 0 &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(r_i\)   | `file_0_view_0` | `Revenue` |
| \(d_i\)   | `file_0_view_0` | `Demand` |
| \(I_i\)   | `file_0_view_0` | `Initial Inventory` |
| \(I\)     | `file_0_view_0` | `Product Name` (filter: contains “Baby”) |

No other data sources are referenced. The model uses only the parameters and indices defined above.