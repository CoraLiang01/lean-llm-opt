**Mathematical Model**

Let  

- \(I\) be the set of pizza types (rows of the table).  
- \(r_i\) – revenue per unit of pizza type \(i\).  
- \(d_i\) – deterministic demand for pizza type \(i\).  
- \(inv_i\) – initial inventory of pizza type \(i\).  
- \(x_i\) – number of units of pizza type \(i\) to be sold (decision variable).

\[
\begin{aligned}
\max \quad & \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\quad
& 0 \le x_i \le d_i && \forall i\in I \\[2pt]
& 0 \le x_i \le inv_i && \forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

```json
{
  "table_id": "file_0_view_0",
  "columns": {
    "Product Name": "i",
    "Revenue": "r_i",
    "Demand": "d_i",
    "Initial Inventory": "inv_i"
  }
}
```

The index set \(I\) is obtained from the distinct values in the **Product Name** column of the table. All parameters \(r_i, d_i, inv_i\) are read directly from the corresponding columns for each \(i\). No other data or constraints are introduced.