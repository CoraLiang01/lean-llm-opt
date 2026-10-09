Let $x_{ij}$ denote the quantity of product shipped from warehouse $i$ to customer (store) $j$.

#### Sets and Parameters (from data):

- Warehouses (indexed by $i$):  
  S1, S2, S3, S4, S5, S6, S7, S8, S9, S10

- Customers (indexed by $j$):  
  C1, C2, C3, C4, C5, C6, C7, C8, C9, C10

- Supply capacities:
  - S1: 127
  - S2: 236
  - S3: 168
  - S4: 115
  - S5: 280
  - S6: 179
  - S7: 135
  - S8: 263
  - S9: 283
  - S10: 476

- Customer demands:
  - C1: 45
  - C2: 23
  - C3: 94
  - C4: 92
  - C5: 57
  - C6: 52
  - C7: 23
  - C8: 99
  - C9: 99
  - C10: 77

- Transportation costs $c_{ij}$ (per unit from warehouse $i$ to customer $j$):

|      |  C1         |  C2         |  C3         |  C4         |  C5         |  C6         |  C7         |  C8         |  C9         |  C10        |
|------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| S1   | 2077.05867  | 0.0         | 54.33526    | 0.0         | 0.0         | 36.17285    | 0.0         | 0.0         | 169.33027   | 0.0         |
| S2   | 2077.05867  | 0.0         | 1141.04056  | 0.0         | 0.0         | 651.11123   | 0.0         | 0.0         | 8.06335     | 0.0         |
| S3   | 79.92103    | 474.24509   | 1477.06763  | 22.58310    | 474.24509   | 41.10660    | 474.24509   | 474.24509   | 624.16254   | 474.24509   |
| S4   | 1659.33693  | 57.20541    | 186.15190   | 1201.31371  | 1029.69746  | 41.82211    | 57.20541    | 1201.31371  | 884.56339   | 1029.69746  |
| S5   | 1297.25670  | 77.76629    | 24.26760    | 1399.79324  | 77.76629    | 53.91162    | 1399.79324  | 77.76629    | 1255.11515  | 1399.79324  |
| S6   | 1998.90907  | 985.31654   | 2.85417     | 1149.53597  | 985.31654   | 730.69236   | 54.73981    | 985.31654   | 46.80310    | 1149.53597  |
| S7   | 1780.33601  | 0.0         | 1141.04056  | 0.0         | 0.0         | 36.17285    | 0.0         | 0.0         | 8.06335     | 0.0         |
| S8   | 75.40936    | 1338.19873  | 21.39135    | 74.34437    | 74.34437    | 937.35062   | 1338.19873  | 1338.19873  | 1392.11866  | 1338.19873  |
| S9   | 98.90756    | 0.0         | 978.03477   | 0.0         | 0.0         | 651.11123   | 0.0         | 0.0         | 169.33027   | 0.0         |
| S10  | 2077.05867  | 0.0         | 54.33526    | 0.0         | 0.0         | 36.17285    | 0.0         | 0.0         | 145.14023   | 0.0         |

#### Decision Variables

- $x_{ij} \geq 0$ : quantity shipped from warehouse $i$ to customer $j$ (continuous, nonnegative)

#### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in \{\text{S1},\ldots,\text{S10}\}} \sum_{j \in \{\text{C1},\ldots,\text{C10}\}} c_{ij} \, x_{ij}
$$

#### Constraints

1. **Supply capacity at each warehouse:**
   $$
   \sum_{j \in \{\text{C1},\ldots,\text{C10}\}} x_{ij} \leq \text{supply\_capacity}_i, \quad \forall i \in \{\text{S1},\ldots,\text{S10}\}
   $$
   - S1: $\sum_j x_{\text{S1},j} \leq 127$
   - S2: $\sum_j x_{\text{S2},j} \leq 236$
   - S3: $\sum_j x_{\text{S3},j} \leq 168$
   - S4: $\sum_j x_{\text{S4},j} \leq 115$
   - S5: $\sum_j x_{\text{S5},j} \leq 280$
   - S6: $\sum_j x_{\text{S6},j} \leq 179$
   - S7: $\sum_j x_{\text{S7},j} \leq 135$
   - S8: $\sum_j x_{\text{S8},j} \leq 263$
   - S9: $\sum_j x_{\text{S9},j} \leq 283$
   - S10: $\sum_j x_{\text{S10},j} \leq 476$

2. **Demand satisfaction at each customer:**
   $$
   \sum_{i \in \{\text{S1},\ldots,\text{S10}\}} x_{ij} = \text{demand}_j, \quad \forall j \in \{\text{C1},\ldots,\text{C10}\}
   $$
   - C1: $\sum_i x_{i,\text{C1}} = 45$
   - C2: $\sum_i x_{i,\text{C2}} = 23$
   - C3: $\sum_i x_{i,\text{C3}} = 94$
   - C4: $\sum_i x_{i,\text{C4}} = 92$
   - C5: $\sum_i x_{i,\text{C5}} = 57$
   - C6: $\sum_i x_{i,\text{C6}} = 52$
   - C7: $\sum_i x_{i,\text{C7}} = 23$
   - C8: $\sum_i x_{i,\text{C8}} = 99$
   - C9: $\sum_i x_{i,\text{C9}} = 99$
   - C10: $\sum_i x_{i,\text{C10}} = 77$

3. **Nonnegativity:**
   $$
   x_{ij} \geq 0 \quad \forall i \in \{\text{S1},\ldots,\text{S10}\},\; j \in \{\text{C1},\ldots,\text{C10}\}
   $$

#### Complete Model

Minimize
$$
\sum_{i \in \{\text{S1},\ldots,\text{S10}\}} \sum_{j \in \{\text{C1},\ldots,\text{C10}\}} c_{ij} \, x_{ij}
$$

Subject to:
$$
\sum_{j} x_{ij} \leq \text{supply\_capacity}_i \quad \forall i
$$
$$
\sum_{i} x_{ij} = \text{demand}_j \quad \forall j
$$
$$
x_{ij} \geq 0 \quad \forall i, j
$$

Where all parameters and indices are as listed above, and $c_{ij}$ are the transportation costs as given in the table.