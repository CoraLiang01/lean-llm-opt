**Mathematical Model**

Let  

* \(S\) – set of suppliers (indices \(i\)),  
* \(C\) – set of customers (indices \(j\)).

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(y_i\) | 1 if supplier \(i\) is opened, 0 otherwise | binary decision variable |
| \(x_{ij}\) | quantity shipped from supplier \(i\) to customer \(j\) | continuous decision variable |
| \(f_i\) | fixed opening cost of supplier \(i\) | parameter |
| \(t_{ij}\) | transportation cost per unit from supplier \(i\) to customer \(j\) | parameter |
| \(d_j\) | demand of customer \(j\) | parameter |

---

### Objective
\[
\min \;\; \sum_{i\in S} f_i\,y_i \;+\; \sum_{i\in S}\sum_{j\in C} t_{ij}\,x_{ij}
\]

### Constraints

1. **Demand satisfaction**  
   \[
   \sum_{i\in S} x_{ij} \;\ge\; d_j \quad \forall j\in C
   \]

2. **Supplier capacity (implicit)** – a supplier can ship only if it is opened  
   \[
   x_{ij} \;\le\; d_j\,y_i \quad \forall i\in S,\; j\in C
   \]

3. **Non‑negativity**  
   \[
   x_{ij} \;\ge\; 0 \quad \forall i\in S,\; j\in C
   \]

4. **Binary opening decisions**  
   \[
   y_i \in \{0,1\} \quad \forall i\in S
   \]

---

**Data Mapping**

```json
{
  "parameters": {
    "f_i": {
      "source_table": "file_1_view_0",
      "row_key": "Unnamed: 0",
      "value_column": "fixed_costs"
    },
    "t_ij": {
      "source_table": "file_2_view_0",
      "row_key": "Unnamed: 0",
      "value_columns": ["C1","C2","C3","C4","C5","C6"]
    },
    "d_j": {
      "source_table": "file_0_view_0",
      "row_key": "customer",
      "value_column": "demand"
    }
  },
  "index_sets": {
    "S": ["S1","S2","S3","S4","S5","S6"],
    "C": ["C1","C2","C3","C4","C5","C6"]
  }
}
```

The model uses the full supplier and customer sets extracted from the three CSV files, with parameters mapped exactly to the corresponding table columns. No additional constraints or bounds are introduced beyond those required by the problem statement.