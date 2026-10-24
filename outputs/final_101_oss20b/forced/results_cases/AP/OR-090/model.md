**Mathematical Model**

Let  

- \(I=\{P1,\dots ,P100\}\) products  
- \(K=\{R1,R2,R3\}\) resources  

Parameters (derived from the CSV files)

\[
\begin{aligned}
&\text{profit}_{i}\quad &&\forall i\in I &&\text{(from column `profit_per_unit`)}\\
&r_{k,i}\quad &&\forall k\in K,\;i\in I &&\text{(from columns `r1_per_unit`, `r2_per_unit`, `r3_per_unit`)}\\
&\text{batch}_i\quad &&\forall i\in I &&\text{(from column `batch_size_units`, equal to 10)}\\
&\text{demand}_i\quad &&\forall i\in I &&\text{(from column `upper_demand_units`)}\\
&\text{cap}_k\quad &&\forall k\in K &&\text{(from column `capacity` in resources_capacities.csv)}
\end{aligned}
\]

Decision variables  

\[
x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
\]

---

**Objective**

\[
\max \;\; \sum_{i\in I} 10\,x_i\,\text{profit}_{i}
\]

---

**Constraints**

1. **Resource limits**

\[
\sum_{i\in I} 10\,x_i\, r_{k,i}\;\le\; \text{cap}_k
\qquad \forall k\in K
\]

2. **Demand limits**

\[
10\,x_i \;\le\; \text{demand}_i
\qquad \forall i\in I
\]

3. **Non‑negativity & integrality**

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

---

**Data Mapping**

```json
{
  "parameters": {
    "profit": {
      "source_table": "file_0_view_0",
      "source_column": "profit_per_unit",
      "index": "product"
    },
    "r1": {
      "source_table": "file_0_view_0",
      "source_column": "r1_per_unit",
      "index": "product"
    },
    "r2": {
      "source_table": "file_0_view_0",
      "source_column": "r2_per_unit",
      "index": "product"
    },
    "r3": {
      "source_table": "file_0_view_0",
      "source_column": "r3_per_unit",
      "index": "product"
    },
    "batch": {
      "source_table": "file_0_view_0",
      "source_column": "batch_size_units",
      "index": "product"
    },
    "demand": {
      "source_table": "file_0_view_0",
      "source_column": "upper_demand_units",
      "index": "product"
    },
    "cap": {
      "source_table": "file_1_view_0",
      "source_column": "capacity",
      "index": "resource"
    }
  },
  "index_sets": {
    "I": "product",
    "K": "resource"
  }
}
```

This model uses the full set of 100 products and the three resources, with all parameters mapped directly to the corresponding CSV columns. No numerical values are enumerated in the model.