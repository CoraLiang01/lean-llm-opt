**Mathematical Model**

Let  

* \(I\) – set of product types (indexed by the values of *ProductName* in *products.csv*).  
* \(W\) – set of warehouses (indexed by the values of *Warehouse ID* in *capacity.csv*).  

Parameters  

* \(v_i\) – benefit (value) of one unit of product \(i\) (from *products.csv*).  
* \(C_w\) – capacity (maximum number of cars) of warehouse \(w\) (from *capacity.csv*).  

Decision variables  

* \(x_{i,w}\) – number of units of product \(i\) stored in warehouse \(w\).  
  \[
  x_{i,w}\in \mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; w\in W
  \]

---

### Objective
\[
\max \;\; \sum_{i\in I}\sum_{w\in W} v_i\,x_{i,w}
\]

### Constraints
1. **Warehouse capacity**  
   \[
   \sum_{i\in I} x_{i,w} \;\le\; C_w \qquad \forall w\in W
   \]

2. **Non‑negativity & integrality**  
   \[
   x_{i,w}\in \mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; w\in W
   \]

---

**Data Mapping**

```json
{
  "parameters": {
    "v_i": {
      "source_table": "file_1_view_0",
      "key_column": "ProductName",
      "value_column": "Value"
    },
    "C_w": {
      "source_table": "file_0_view_0",
      "key_column": "Warehouse ID",
      "value_column": "Capacity"
    }
  },
  "variables": {
    "x_{i,w}": {
      "domain": "integer non‑negative",
      "indices": ["ProductName", "Warehouse ID"]
    }
  }
}
```

*The model uses only the data explicitly provided in the two CSV files; no additional assumptions or numerical values are introduced.*