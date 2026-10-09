##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity of beverages shipped from plant $i$ to retail outlet $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (production plants)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (retail outlets)

##### Parameters

- $d_j$: demand at outlet $j$ (from customer_demand.csv)
- $s_i$: supply capacity at plant $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from plant $i$ to outlet $j$ (from transportation_costs.csv)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:** Each outlet's demand must be met:
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
2. **Supply capacity:** Each plant's shipments cannot exceed its capacity:
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
3. **Non-negativity:**
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Data Mapping

- $I$ (plants): S1, S2, S3, S4 (from supply_capacity.csv and transportation_costs.csv, column "supplier_id")
- $J$ (outlets): C1, C2, C3, C4 (from customer_demand.csv, column "customer_id" and transportation_costs.csv, columns "transportation_cost_to_C1", etc.)
- $d_j$ (demand): from customer_demand.csv, column "demand", indexed by "customer_id"
- $s_i$ (capacity): from supply_capacity.csv, column "supply_capacity", indexed by "supplier_id"
- $c_{ij}$ (cost): from transportation_costs.csv, value in row with "supplier_id" $i$ and column "transportation_cost_to_{j}$

All indices, parameters, and coefficients are mapped directly from the provided CSV files as described above.