**Mathematical Model**

Let  

- \(I\) be the set of real‑estate areas (rows of *products.csv*).  
- \(C\) be the overall development capacity (value in *capacity.csv*).  
- \(v_i\) be the benefit per unit of development in area \(i\) (column *Value*).  
- \(w_i\) be the resource consumption per unit of development in area \(i\) (column *Weight*).  
- \(x_i\) be the scale of development per day in area \(i\).

\[
\begin{aligned}
\max_{x}\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t. } & \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \;\ge\; 0 \qquad \forall i\in I
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
        "ProductName": "i",
        "Value": "v_i",
        "Weight": "w_i"
      }
    }
  ]
}
```

*No numerical values are enumerated; all parameters are linked to the source data as shown.*