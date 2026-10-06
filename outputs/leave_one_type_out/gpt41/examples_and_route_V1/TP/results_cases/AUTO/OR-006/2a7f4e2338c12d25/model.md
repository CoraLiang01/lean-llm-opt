#### Sets

- $S$: set of warehouses, indexed by $s$ (from supply_capacity.csv: S1, S2, ..., S10)
- $C$: set of customers (stores), indexed by $c$ (from customer_demand.csv: C1, C2, ..., C10)

#### Parameters

- $demand_c$: daily demand for customer $c$ (from customer_demand.csv)
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

- $supply\_capacity_s$: daily supply capacity of warehouse $s$ (from supply_capacity.csv)
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

- $cost_{s,c}$: cost to transport one unit from warehouse $s$ to customer $c$ (from transportation_costs.csv):

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

- $x_{s,c}$: number of units shipped from warehouse $s$ to customer $c$; $x_{s,c} \geq 0$ and integer

#### Mathematical Model

Minimize total transportation cost:
$$
\min \sum_{s \in S} \sum_{c \in C} cost_{s,c} \cdot x_{s,c}
$$

Subject to:

1. **Demand satisfaction for each customer:**
   $$
   \sum_{s \in S} x_{s,c} = demand_c, \quad \forall c \in C
   $$

2. **Supply capacity for each warehouse:**
   $$
   \sum_{c \in C} x_{s,c} \leq supply\_capacity_s, \quad \forall s \in S
   $$

3. **Nonnegativity and integrality:**
   $$
   x_{s,c} \in \mathbb{Z}_{\geq 0}, \quad \forall s \in S, \forall c \in C
   $$

#### Data Used

- Warehouses: S1, S2, S3, S4, S5, S6, S7, S8, S9, S10
- Customers: C1, C2, C3, C4, C5, C6, C7, C8, C9, C10
- Demands: as listed above
- Supply capacities: as listed above
- Transportation costs: as listed above (see table)

This model determines the optimal integer shipment plan from each warehouse to each customer, minimizing total cost, meeting all customer demands, and not exceeding any warehouse's supply capacity.