**Mathematical Model**

Let  

* \(I=\{S_1,S_2,\dots ,S_{24}\}\) – set of suppliers  
* \(J=\{C_1,C_2,\dots ,C_{25}\}\) – set of supermarkets  

Parameters  

* \(f_i\) – fixed opening cost of supplier \(i\)  \(i\in I\)  
* \(d_j\) – demand of supermarket \(j\)  \(j\in J\)  
* \(t_{ij}\) – transportation cost per unit from supplier \(i\) to supermarket \(j\) \(i\in I,\;j\in J\)

Decision variables  

* \(y_i\in\{0,1\}\) – 1 if supplier \(i\) is opened, 0 otherwise  
* \(x_{ij}\ge 0\) – quantity shipped from supplier \(i\) to supermarket \(j\)

---

### Objective
\[
\min \;\; \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} t_{ij}\,x_{ij}
\]

### Constraints  

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij} \;=\; d_j \qquad \forall j\in J
   \]

2. **Linking open supplier and shipments**  
   \[
   x_{ij} \;\le\; d_j\,y_i \qquad \forall i\in I,\; j\in J
   \]

3. **Variable domains**  
   \[
   y_i \in \{0,1\}\quad \forall i\in I,\qquad
   x_{ij}\ge 0\quad \forall i\in I,\; j\in J
   \]

---

**Data Mapping**

```json
{
  "parameters": {
    "f_i": {
      "source_table": "file_1_view_0",
      "row_id_column": "Unnamed: 0",
      "value_column": "fixed_costs",
      "index": "I"
    },
    "d_j": {
      "source_table": "file_0_view_0",
      "row_id_column": "customer",
      "value_column": "demand",
      "index": "J"
    },
    "t_ij": {
      "source_table": "file_2_view_0",
      "row_id_column": "Unnamed: 0",
      "column_id_column": "C1",  // all columns C1..C25
      "value_column": "C1",      // value per column
      "index_i": "I",
      "index_j": "J"
    }
  },
  "variables": {
    "y_i": {
      "type": "binary",
      "index": "I"
    },
    "x_ij": {
      "type": "continuous",
      "lower_bound": 0,
      "index_i": "I",
      "index_j": "J"
    }
  }
}
```

*The mapping specifies that each parameter is read from the corresponding CSV view, with supplier identifiers in the `Unnamed: 0` column and customer identifiers in the `customer` column. The transportation cost matrix is taken row‑wise from `file_2_view_0`, where each row corresponds to a supplier and each column `C1`–`C25` to a supermarket.*