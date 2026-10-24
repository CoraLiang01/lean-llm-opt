**Mathematical Model**

Let  

- \(I\) be the set of dairy products indexed by the values in the column **Full_Product_Name**.  
- For each \(i\in I\)  
  - \(r_i\) : revenue per unit (from column **Revenue**)  
  - \(d_i\) : demand quantity (from column **Demand**)  
  - \(inv_i\) : initial inventory (from column **Initial Inventory**)  
  - \(x_i\) : number of units of product \(i\) to be fulfilled  

\[
\begin{aligned}
\max\quad & \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\quad
& 0 \le x_i \le d_i && \forall i\in I \\[2pt]
& 0 \le x_i \le inv_i && \forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{+} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

```json
{
  "table_id": "file_0_view_0",
  "index_set": {
    "I": {
      "source_column": "Full_Product_Name",
      "type": "string"
    }
  },
  "parameters": {
    "r": {
      "source_column": "Revenue",
      "type": "float"
    },
    "d": {
      "source_column": "Demand",
      "type": "int"
    },
    "inv": {
      "source_column": "Initial Inventory",
      "type": "int"
    }
  },
  "variables": {
    "x": {
      "domain": "integer",
      "bounds": {
        "lower": 0,
        "upper": "min(d_i, inv_i)"
      }
    }
  }
}
```

The model uses the full set of products present in the CSV file; no additional filtering or aggregation is applied.