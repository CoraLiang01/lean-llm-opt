#### Index Sets

- $I$: set of plants (from cost.csv, column plant)
- $J$: set of customers (from demand.csv, column customer)

#### Parameters

- $f_i$: fixed opening cost for plant $i \in I$ (cost.csv, column fixed_cost)
- $K_i$: capacity of plant $i \in I$ (cost.csv, column capacity)
- $c_{ij}$: per-unit transport cost from plant $i$ to customer $j$ ($i \in I$, $j \in J$) (cost.csv, columns C1–C15)
- $d_j$: demand of customer $j \in J$ (demand.csv, column demand)

#### Decision Variables

- $y_i \in \{0,1\}$: 1 if plant $i$ is built (opened), 0 otherwise
- $x_{ij} \geq 0$: quantity shipped from plant $i$ to customer $j$ (continuous, nonnegative)

#### Objective

Minimize total cost (fixed opening + transportation):
$$
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

#### Constraints

1. **Demand Satisfaction** (every customer’s demand must be met):
   $$
   \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J
   $$

2. **Plant Capacity** (cannot ship more than capacity from any plant):
   $$
   \sum_{j \in J} x_{ij} \leq K_i y_i \quad \forall i \in I
   $$

3. **Variable Domains**:
   $$
   y_i \in \{0,1\} \quad \forall i \in I
   $$
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $I$ (plants): file_0_view_0, column plant (cost.csv)
- $J$ (customers): file_1_view_0, column customer (demand.csv)
- $f_i$: file_0_view_0, column fixed_cost (cost.csv)
- $K_i$: file_0_view_0, column capacity (cost.csv)
- $c_{ij}$: file_0_view_0, columns C1–C15 (cost.csv), for each $i$ and $j$
- $d_j$: file_1_view_0, column demand (demand.csv)

All rows from both files are used as returned by CSVQA.