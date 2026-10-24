**Mathematical Model**

Let  

- \(I\) be the set of all products (identified by *Product Name*).  
- \(r_i\) – revenue per unit of product \(i\).  
- \(d_i\) – demand quantity of product \(i\).  
- \(s_i\) – initial inventory of product \(i\).  
- \(x_i\) – quantity of product \(i\) to sell (decision variable).

\[
\begin{aligned}
\max \quad & \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\quad
& 0 \le x_i \le d_i && \forall i\in I \\[2pt]
& 0 \le x_i \le s_i && \forall i\in I \\[2pt]
& x_i \in \mathbb{R} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

```json
{
  "table_id": "file_0_view_0",
  "columns": {
    "Product Name": "I",
    "Revenue": "r_i",
    "Demand": "d_i",
    "Initial Inventory": "s_i"
  }
}
```

*All parameters \(r_i, d_i, s_i\) are taken directly from the corresponding columns of the table `RetailSalesDataset.csv`.*