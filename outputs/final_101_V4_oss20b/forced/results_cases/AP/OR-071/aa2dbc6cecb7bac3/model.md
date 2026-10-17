**Mathematical Model**

Let  

- \(P\) be the set of all products (identified by *Product Name*).  
- \(x_p\) be the weekly production quantity of product \(p\) \((x_p\ge 0)\).  

Parameters (obtained from *41.csv*)  

- \(l_p\) : labor required per unit of product \(p\)  \((\text{Labor per unit})\).  
- \(m_p\) : material required per unit of product \(p\)  \((\text{Material per unit})\).  
- \(s_p\) : selling price per unit of product \(p\)  \((\text{Selling Price})\).  
- \(c_p\) : variable cost per unit of product \(p\)  \((\text{Variable Cost})\).  

Fixed weekly operating cost: \(F = 4500\).

\[
\begin{aligned}
\max_{x}\quad & \sum_{p\in P} (s_p-c_p)\,x_p \;-\; F \\[4pt]
\text{s.t.}\quad & \sum_{p\in P} l_p\,x_p \;\le\; 1650 \\[4pt]
& \sum_{p\in P} m_p\,x_p \;\le\; 1850 \\[4pt]
& x_p \;\ge\; 0 \qquad \forall p\in P
\end{aligned}
\]

---

**Data Mapping**

```json
{
  "table_id": "file_0_view_0",
  "index_set": {
    "P": {
      "source_column": "Product Name",
      "type": "string"
    }
  },
  "parameters": {
    "l": {
      "source_column": "Labor per unit",
      "type": "float"
    },
    "m": {
      "source_column": "Material per unit",
      "type": "float"
    },
    "s": {
      "source_column": "Selling Price",
      "type": "float"
    },
    "c": {
      "source_column": "Variable Cost",
      "type": "float"
    }
  },
  "fixed_cost": 4500,
  "capacities": {
    "labor": 1650,
    "material": 1850
  }
}
```

The model uses the full set of 198 products from the CSV; no values are enumerated in the formulation.