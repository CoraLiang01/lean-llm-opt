**Mathematical Model**

Let  

- \(I\) be the set of development areas (rows of *products.csv*).  
- For each \(i\in I\)  
  - \(b_i\) = benefit coefficient (column **Value**).  
  - \(w_i\) = development‑unit weight (column **Weight**).  
- \(C\) = overall capacity (column **Capacity** in *capacity.csv*).  

Decision variables  

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

Objective  

\[
\max \;\sum_{i\in I} b_i\,x_i
\]

Capacity constraint  

\[
\sum_{i\in I} w_i\,x_i \;\le\; C
\]

---

**Data Mapping**

```json
{
  "tables": [
    {
      "table_id": "file_1_view_0",
      "role": "descriptive non-unique role",
      "columns": ["ProductName", "Value", "Weight"],
      "index": "ProductName",
      "parameters": {
        "b": "Value",
        "w": "Weight"
      }
    },
    {
      "table_id": "file_0_view_0",
      "role": "descriptive non-unique role",
      "columns": ["Capacity"],
      "parameters": {
        "C": "Capacity"
      }
    }
  ]
}
```

This mapping binds each symbolic parameter to the exact source column values without enumerating the numeric table.