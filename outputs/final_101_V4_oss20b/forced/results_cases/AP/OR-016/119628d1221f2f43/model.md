**Mathematical Model**

Let  

- \(I\) be the set of all products (identified by *Product Name*).  
- For each \(i\in I\)  
  - \(r_i\) : revenue per unit (from column **Revenue**)  
  - \(d_i\) : demand quantity (from column **Demand**)  
  - \(s_i\) : initial inventory (from column **Initial Inventory**)  

Decision variable  

- \(x_i\) : quantity of product \(i\) to sell (continuous, non‑negative).

\[
\begin{aligned}
\max_{x}\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. }& 0 \le x_i \le d_i && \forall i\in I \\[2pt]
& 0 \le x_i \le s_i && \forall i\in I
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

This mapping links each column of the CSV file *RetailSalesDataset.csv* to the corresponding index set or parameter in the model. No numerical values are enumerated; the model remains symbolic and ready for numerical instantiation.