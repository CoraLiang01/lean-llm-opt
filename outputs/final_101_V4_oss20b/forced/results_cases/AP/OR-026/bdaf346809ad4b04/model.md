**Mathematical Model**

Let  

- \(I\) – set of all products whose *Product Name* contains the word “Fashion” (extracted from the filtered table).  
- For each \(i\in I\)  
  - \(r_i\) – revenue per unit (from column **Revenue**).  
  - \(d_i\) – deterministic demand (from column **Demand**).  
  - \(v_i\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

- \(x_i\) – number of units of product \(i\) to be sold (fulfilled).  

\[
\begin{aligned}
\max_{x}\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le \min\{d_i,\;v_i\}\quad \forall i\in I \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
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
        "column": "Product Name",
        "operator": "contains",
        "value": "Fashion"
      }
    }
  },
  "parameters": {
    "r": {
      "source_column": "Revenue",
      "type": "numeric"
    },
    "d": {
      "source_column": "Demand",
      "type": "numeric"
    },
    "v": {
      "source_column": "Initial Inventory",
      "type": "numeric"
    }
  },
  "variables": {
    "x": {
      "domain": "integer",
      "bounds": {
        "lower": 0,
        "upper": "min(d_i, v_i)"
      }
    }
  }
}
```

*All parameter values are taken directly from the filtered rows of the CSV file; no numerical values are enumerated in the model.*