#### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups, as defined by the retrieved data.

**Decision Variables:**

For each $i \in I$, $j \in J$:
$$
x_{ij} \geq 0
$$
where $x_{ij}$ is the quantity shipped from distribution center $i$ to customer group $j$ (continuous).

**Parameters:**

- $d_j$: demand of customer group $j$ (from "customer_demand.csv")
- $s_i$: supply capacity of distribution center $i$ (from "supply_capacity.csv")
- $c_{ij}$: transportation cost per unit from $i$ to $j$ (from "transportation_costs.csv")

**Objective:**

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

**Constraints:**

1. **Demand satisfaction:**  
   For each customer group $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
2. **Supply capacity:**  
   For each distribution center $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

#### Data Mapping

- $I$ (distribution centers): all supplier_id in "supply_capacity.csv" and "transportation_costs.csv" (S1, S2, ..., S18)
- $J$ (customer groups): all customer_id in "customer_demand.csv" and "transportation_costs.csv" (C1, C2, ..., C18)
- $d_j$: "demand_units" column in "customer_demand.csv", indexed by "customer_id"
- $s_i$: "supply_capacity_units" column in "supply_capacity.csv", indexed by "supplier_id"
- $c_{ij}$: "transportation_cost_to_Ck" columns in "transportation_costs.csv", with row "supplier_id" $i$ and column for customer $j$ (see relationships in the Observation)

**Index sets and all coefficients are defined by the full, unsimplified, and source-ordered data as returned above.**