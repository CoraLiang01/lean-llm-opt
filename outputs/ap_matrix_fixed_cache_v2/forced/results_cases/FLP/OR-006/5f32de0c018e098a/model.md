##### Decision Variables

$x_{ij} \geq 0$: quantity of product shipped from warehouse $i \in I$ to store $j \in J$ (continuous, nonnegative).

##### Parameters

- Warehouses $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$
- Stores $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$

- Store demands $d_j$:
  - $d_{\text{C1}} = 45$
  - $d_{\text{C2}} = 23$
  - $d_{\text{C3}} = 94$
  - $d_{\text{C4}} = 92$
  - $d_{\text{C5}} = 57$
  - $d_{\text{C6}} = 52$
  - $d_{\text{C7}} = 23$
  - $d_{\text{C8}} = 99$
  - $d_{\text{C9}} = 99$
  - $d_{\text{C10}} = 77$

- Warehouse supply capacities $s_i$:
  - $s_{\text{S1}} = 127$
  - $s_{\text{S2}} = 236$
  - $s_{\text{S3}} = 168$
  - $s_{\text{S4}} = 115$
  - $s_{\text{S5}} = 280$
  - $s_{\text{S6}} = 179$
  - $s_{\text{S7}} = 135$
  - $s_{\text{S8}} = 263$
  - $s_{\text{S9}} = 283$
  - $s_{\text{S10}} = 476$

- Transportation costs $c_{ij}$ (per unit from warehouse $i$ to store $j$):

|        |  C1         |  C2         |  C3         |  C4         |  C5         |  C6         |  C7         |  C8         |  C9         |  C10        |
|--------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| S1     | 2077.05867  | 0.0         | 54.33526    | 0.0         | 0.0         | 36.17285    | 0.0         | 0.0         | 169.33027   | 0.0         |
| S2     | 2077.05867  | 0.0         | 1141.04056  | 0.0         | 0.0         | 651.11123   | 0.0         | 0.0         | 8.06335     | 0.0         |
| S3     | 79.92103    | 474.24509   | 1477.06763  | 22.58310    | 474.24509   | 41.10660    | 474.24509   | 474.24509   | 624.16254   | 474.24509   |
| S4     | 1659.33693  | 57.20541    | 186.15190   | 1201.31371  | 1029.69746  | 41.82211    | 57.20541    | 1201.31371  | 884.56339   | 1029.69746  |
| S5     | 1297.25670  | 77.76629    | 24.26760    | 1399.79324  | 77.76629    | 53.91162    | 1399.79324  | 77.76629    | 1255.11515  | 1399.79324  |
| S6     | 1998.90907  | 985.31654   | 2.85417     | 1149.53597  | 985.31654   | 730.69236   | 54.73981    | 985.31654   | 46.80310    | 1149.53597  |
| S7     | 1780.33601  | 0.0         | 1141.04056  | 0.0         | 0.0         | 36.17285    | 0.0         | 0.0         | 8.06335     | 0.0         |
| S8     | 75.40936    | 1338.19873  | 21.39135    | 74.34437    | 74.34437    | 937.35062   | 1338.19873  | 1338.19873  | 1392.11866  | 1338.19873  |
| S9     | 98.90756    | 0.0         | 978.03477   | 0.0         | 0.0         | 651.11123   | 0.0         | 0.0         | 169.33027   | 0.0         |
| S10    | 2077.05867  | 0.0         | 54.33526    | 0.0         | 0.0         | 36.17285    | 0.0         | 0.0         | 145.14023   | 0.0         |

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction for each store:**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supply capacity for each warehouse:**
   \[
   \sum_{j \in J} x_{ij} \leq s_i, \quad \forall i \in I
   \]

3. **Nonnegativity:**
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]

##### Retrieved Information

- Warehouses: S1, S2, S3, S4, S5, S6, S7, S8, S9, S10
- Stores: C1, C2, C3, C4, C5, C6, C7, C8, C9, C10
- Demands: as listed above
- Supply capacities: as listed above
- Transportation costs: as listed above (matrix form)

This is a classical transportation problem with supply and demand constraints and nonnegative continuous shipment variables.