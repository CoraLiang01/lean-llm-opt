**Mathematical Model**

Let  

* \(S\) – set of suppliers (indexed by \(s\))  
* \(C\) – set of customers (indexed by \(c\))

| Symbol | Meaning |
|--------|---------|
| \(y_s\) | Binary variable, 1 if supplier \(s\) is opened, 0 otherwise |
| \(x_{s c}\) | Quantity of product shipped from supplier \(s\) to customer \(c\) |
| \(f_s\) | Fixed opening cost of supplier \(s\) |
| \(t_{s c}\) | Transportation cost per unit from supplier \(s\) to customer \(c\) |
| \(d_c\) | Demand of customer \(c\) |

\[
\begin{aligned}
\min\;& \sum_{s\in S} f_s\,y_s \;+\; \sum_{s\in S}\sum_{c\in C} t_{s c}\,x_{s c} \\[4pt]
\text{s.t.}\;& \sum_{s\in S} x_{s c} \;\ge\; d_c \qquad &&\forall c\in C \\[4pt]
& x_{s c} \;\le\; d_c\,y_s \qquad &&\forall s\in S,\; \forall c\in C \\[4pt]
& x_{s c} \;\ge\; 0 \qquad &&\forall s\in S,\; \forall c\in C \\[4pt]
& y_s \in \{0,1\} \qquad &&\forall s\in S
\end{aligned}
\]

---

**Data Mapping**

```json
{
  "index_sets": {
    "S": {
      "table_id": "file_1_view_0",
      "column": "Unnamed: 0",
      "role": "supplier_id"
    },
    "C": {
      "table_id": "file_0_view_0",
      "column": "customer",
      "role": "customer_id"
    }
  },
  "parameters": {
    "f_s": {
      "table_id": "file_1_view_0",
      "column": "fixed_costs",
      "index": "S"
    },
    "d_c": {
      "table_id": "file_0_view_0",
      "column": "demand",
      "index": "C"
    },
    "t_s_c": {
      "table_id": "file_2_view_0",
      "columns": ["C1","C2","C3","C4","C5","C6"],
      "index": ["S","C"],
      "row_key": "Unnamed: 0",
      "column_key": "customer"
    }
  }
}
```

* `S` is populated from the `Unnamed: 0` column of `fixed_cost.csv`.  
* `C` is populated from the `customer` column of `demand.csv`.  
* `f_s` maps each supplier to its fixed cost.  
* `d_c` maps each customer to its demand.  
* `t_s_c` maps each supplier‑customer pair to the corresponding transportation cost from `transportation_costs.csv`.