**Mathematical Model**

Let  

- \(I\) be the set of product identifiers that satisfy the filter  
  (`id_number` starts with *id999*).  
- For each \(i\in I\)  
  - \(r_i\) – revenue per unit (from column **Revenue**)  
  - \(d_i\) – demand during the horizon (from column **Demand**)  
  - \(v_i\) – initial inventory (from column **Initial Inventory**)  
- Decision variable  
  - \(x_i\) – units of product \(i\) to fulfill (integer, \(\ge 0\))

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & x_i \;\le\; d_i && \forall i\in I \\[2pt]
             & x_i \;\le\; v_i && \forall i\in I \\[2pt]
             & x_i \in \mathbb{Z}_{+} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(r_i\)   | `file_0_view_0` | `Revenue` |
| \(d_i\)   | `file_0_view_0` | `Demand` |
| \(v_i\)   | `file_0_view_0` | `Initial Inventory` |
| \(I\)     | `file_0_view_0` | `id_number` (filtered by prefix `id999`) |

The table `file_0_view_0` contains the single returned record:

```json
{
  "id_number": "id999",
  "Revenue": "434.74",
  "Demand": "8171",
  "Initial Inventory": "56450"
}
```

All parameters are mapped directly to these columns; no numerical values are enumerated in the model.