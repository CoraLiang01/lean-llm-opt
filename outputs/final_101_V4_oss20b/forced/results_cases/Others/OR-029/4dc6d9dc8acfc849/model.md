**Mathematical Model**

Let  

- \(I\) be the set of FAUX products (rows returned by the filter).  
- For each \(i\in I\)  
  - \(r_i\)  – revenue per unit (yuan)  
  - \(d_i\)  – deterministic demand (units)  
  - \(s_i\)  – initial inventory (units)  

Decision variable  

- \(x_i\in \mathbb{Z}_{+}\) – number of units of product \(i\) to sell.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& 0 \le x_i \le d_i &&\forall i\in I \\[2pt]
& 0 \le x_i \le s_i &&\forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{+} &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Table ID | Columns Used | Filter Condition | Index Set | Parameter Mapping |
|----------|--------------|------------------|-----------|-------------------|
| `file_0_view_0` | `Product Name`, `Revenue`, `Demand`, `Initial Inventory` | `Product Name` contains `"FAUX"` | \(I\) = set of returned rows | \(r_i \leftarrow\) `Revenue`<br>\(d_i \leftarrow\) `Demand`<br>\(s_i \leftarrow\) `Initial Inventory` |

No other tables or parameters are referenced. The model uses only the filtered FAUX product data, respecting demand and inventory limits while maximizing total revenue.