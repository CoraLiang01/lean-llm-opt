**Mathematical Model**

Let  

- \(I\) be the set of shelves indexed by the values in column **ShelfID** of *capacity.csv*.  
- \(J\) be the set of products indexed by the values in column **ProductName** of *products.csv*.  

Parameters  

- \(c_i\) – capacity of shelf \(i\) (units of weight).  
- \(v_j\) – value per unit of product \(j\).  
- \(w_j\) – weight per unit of product \(j\).  

Decision variables  

- \(x_{ij}\) – number of units of product \(j\) placed on shelf \(i\).  

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{j\in J} w_j\,x_{ij} \;\le\; c_i \qquad \forall i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{+}\qquad \forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

```json
{
  "parameters": {
    "c_i": {
      "source_table": "file_0_view_0",
      "row_key": "ShelfID",
      "value_column": "Capacity"
    },
    "v_j": {
      "source_table": "file_1_view_0",
      "row_key": "ProductName",
      "value_column": "Value"
    },
    "w_j": {
      "source_table": "file_1_view_0",
      "row_key": "ProductName",
      "value_column": "Weight"
    }
  },
  "index_sets": {
    "I": {
      "source_table": "file_0_view_0",
      "key_column": "ShelfID"
    },
    "J": {
      "source_table": "file_1_view_0",
      "key_column": "ProductName"
    }
  }
}
```

This mapping links each symbolic parameter and index set directly to the corresponding columns in the provided CSV files. No numerical values are enumerated in the model.