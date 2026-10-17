**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \text{rev}_i\,x_i \\[4pt]
\text{subject to}\quad
& 0 \le x_i \le \text{inv}_i && \forall i\in I \\[2pt]
& 0 \le x_i \le \text{dem}_i && \forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_+ && \forall i\in I
\end{aligned}
\]

**Index Sets**

* \(I\) – set of all product SKUs that start with the prefix “ZZ”.

**Parameters**

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(\text{rev}_i\) | Revenue per unit of product \(i\) | `file_0_view_0` column **Revenue** |
| \(\text{dem}_i\) | Demand quantity of product \(i\) | `file_0_view_0` column **Demand** |
| \(\text{inv}_i\) | Initial inventory of product \(i\) | `file_0_view_0` column **Initial Inventory** |

**Decision Variables**

* \(x_i\) – number of units of product \(i\) to fulfill (integer, non‑negative).

---

**Data Mapping**

```json
{
  "table_id": "file_0_view_0",
  "columns": ["SKU", "Revenue", "Demand", "Initial Inventory"],
  "index_set": {
    "I": {
      "source_column": "SKU",
      "filter": {"prefix": "ZZ"}
    }
  },
  "parameters": {
    "rev": {"source_column": "Revenue"},
    "dem": {"source_column": "Demand"},
    "inv": {"source_column": "Initial Inventory"}
  }
}
```

This mapping links each parameter and the index set directly to the corresponding columns in the source CSV, ensuring the model uses the exact data provided.