#### Index Sets

- $I$: set of plants (from cost.csv, column "plant")
- $J$: set of customers (from demand.csv, column "customer")

#### Parameters

- $f_i$: fixed opening cost for plant $i \in I$ (cost.csv, column "fixed_cost")
- $K_i$: capacity of plant $i \in I$ (cost.csv, column "capacity")
- $c_{ij}$: per-unit transport cost from plant $i$ to customer $j$ (cost.csv, columns "C1"–"C15")
- $d_j$: demand of customer $j \in J$ (demand.csv, column "demand")

#### Decision Variables

- $y_i \in \{0,1\}$: 1 if plant $i$ is built (opened), 0 otherwise
- $x_{ij} \geq 0$: amount shipped from plant $i$ to customer $j$

#### Objective

Minimize total cost (fixed opening costs plus transportation costs):
$$
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

#### Constraints

1. **Demand Satisfaction** (each customer's demand must be met):
   $$
   \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J
   $$

2. **Plant Capacity** (do not exceed capacity of any opened plant):
   $$
   \sum_{j \in J} x_{ij} \leq K_i y_i \quad \forall i \in I
   $$

3. **Non-negativity and Binary Constraints**:
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$
   $$
   y_i \in \{0,1\} \quad \forall i \in I
   $$

---

#### Data Mapping

- **cost.csv** (`file_0_view_0`):
  - Plants: index set $I$ from column "plant"
  - Fixed opening cost $f_i$: column "fixed_cost"
  - Capacity $K_i$: column "capacity"
  - Per-unit transport cost $c_{ij}$: columns "C1"–"C15" (for each $i \in I$, $j \in J$)
- **demand.csv** (`file_1_view_0`):
  - Customers: index set $J$ from column "customer"
  - Demand $d_j$: column "demand"

No additional filtering or selection was applied; all rows and columns from both files are included as per the query.