**Mathematical Model**

Let  

- \(I\) be the set of products whose *Product Name* starts with **S700_**.  
- For each \(i\in I\)  
  - \(r_i\) : revenue per unit (from column **Revenue**)  
  - \(d_i\) : demand quantity (from column **Demand**)  
  - \(v_i\) : initial inventory (from column **Initial Inventory**)  
- Decision variable  
  - \(x_i\) : number of units of product \(i\) to sell (integer, \(\ge 0\))

\[
\begin{aligned}
\max \quad & \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\quad
& 0 \le x_i \le d_i && \forall i\in I \\[2pt]
& 0 \le x_i \le v_i && \forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

```json
{
  "table_id": "file_0_view_0",
  "index_set": {
    "I": {
      "source_column": "Product Name",
      "filter": {
        "prefix": "S700_"
      }
    }
  },
  "parameters": {
    "r": {
      "source_column": "Revenue"
    },
    "d": {
      "source_column": "Demand"
    },
    "v": {
      "source_column": "Initial Inventory"
    }
  },
  "variables": {
    "x": {
      "domain": "integer",
      "bounds": {
        "lower": 0
      }
    }
  }
}
```

This mapping links each symbolic element of the model to the corresponding column in the CSV file `SampleSalesData.csv`. No numerical values are enumerated; the model remains fully symbolic.