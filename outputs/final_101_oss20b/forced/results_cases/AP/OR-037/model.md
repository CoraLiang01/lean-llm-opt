**Mathematical Model**

Let  

- \(I\) be the set of product types (rows of *products.csv*).  
- \(p_i\) be the profit per unit of product \(i\) (column **Value**).  
- \(w_i\) be the weight (inventory space) per unit of product \(i\) (column **Weight**).  
- \(C\) be the total inventory capacity (column **Capacity**).  
- \(x_i\) be the number of units of product \(i\) ordered per day.

\[
\begin{aligned}
\max \quad & \sum_{i\in I} p_i\,x_i \\[4pt]
\text{s.t.}\quad & \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \;\ge\; 0 \quad \forall i\in I \\[4pt]
& x_i \;\in\; \mathbb{Z} \quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

```json
{
  "tables": [
    {
      "table_id": "file_0_view_0",
      "source_file": "capacity.csv",
      "columns": ["Capacity"],
      "parameter": "C",
      "mapping": {
        "Capacity": "C"
      }
    },
    {
      "table_id": "file_1_view_0",
      "source_file": "products.csv",
      "columns": ["ProductName", "Value", "Weight"],
      "index_set": "I",
      "parameters": {
        "p_i": "Value",
        "w_i": "Weight"
      },
      "mapping": {
        "ProductName": "i",
        "Value": "p_i",
        "Weight": "w_i"
      }
    }
  ]
}
```

*All parameters are taken directly from the corresponding CSV columns; no numerical values are enumerated in the model.*