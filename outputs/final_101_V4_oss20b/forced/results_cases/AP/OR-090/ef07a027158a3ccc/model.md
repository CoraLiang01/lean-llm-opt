##### Model

**Indices**  
\(I=\{P1,\dots ,P100\}\) products  
\(R=\{R1,R2,R3\}\) resources  

**Parameters**  
\(profit_i\) profit per unit of product \(i\)  
\(r_{1,i}, r_{2,i}, r_{3,i}\) resource consumption per unit of product \(i\)  
\(upper_i\) maximum market demand (units) for product \(i\)  
\(batch_i\) batch size in units (10) for product \(i\)  
\(capacity_r\) total available amount of resource \(r\)

**Decision variables**  
\(x_i \in \mathbb{Z}_{+}\) number of batches produced of product \(i\)

**Objective**  
\[
\max \;\sum_{i\in I} 10\,x_i\,profit_i
\]

**Constraints**

1. **Resource limits**  
\[
\sum_{i\in I} 10\,x_i\,r_{k,i}\;\le\; capacity_k
\quad \forall k\in R
\]

2. **Demand limits**  
\[
10\,x_i \;\le\; upper_i
\quad \forall i\in I
\]

3. **Non‑negativity & integrality**  
\[
x_i \in \mathbb{Z}_{+}\quad \forall i\in I
\]

---

##### Data Mapping

```json
{
  "parameters": {
    "profit_i": {
      "table_id": "file_0_view_0",
      "column": "profit_per_unit",
      "index": "product"
    },
    "r1_i": {
      "table_id": "file_0_view_0",
      "column": "r1_per_unit",
      "index": "product"
    },
    "r2_i": {
      "table_id": "file_0_view_0",
      "column": "r2_per_unit",
      "index": "product"
    },
    "r3_i": {
      "table_id": "file_0_view_0",
      "column": "r3_per_unit",
      "index": "product"
    },
    "upper_i": {
      "table_id": "file_0_view_0",
      "column": "upper_demand_units",
      "index": "product"
    },
    "batch_i": {
      "table_id": "file_0_view_0",
      "column": "batch_size_units",
      "index": "product"
    },
    "capacity_r": {
      "table_id": "file_1_view_0",
      "column": "capacity",
      "index": "resource"
    }
  }
}
```