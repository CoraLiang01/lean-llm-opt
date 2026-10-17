**Mathematical Model**

Let  

- \(I\) be the set of product types.  
- For each \(i\in I\):  
  - \(v_i\) = benefit coefficient (from *products.csv* column **Value**).  
  - \(w_i\) = weight/space requirement (from *products.csv* column **Weight**).  
- \(C\) = total inventory capacity (from *capacity.csv* column **Capacity**).  
- Decision variable \(x_i\in \mathbb{Z}_{\ge 0}\) = units of product \(i\) ordered daily.

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t. } & \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
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
        "v_i": "Value",
        "w_i": "Weight"
      },
      "index_mapping": {
        "ProductName": "i"
      }
    }
  ]
}
```
