##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center (supplier) $i \in I$ to customer group $j \in J$.

##### Sets

- $I$ = set of supplier IDs from "supply_capacity.csv":  
  $I = \{\text{S1}, \text{S2}, \ldots, \text{S18}\}$
- $J$ = set of customer IDs from "customer_demand.csv":  
  $J = \{\text{C1}, \text{C2}, \ldots, \text{C18}\}$

##### Parameters

- $d_j$ = demand (units) of customer $j \in J$ from "customer_demand.csv"
- $s_i$ = supply capacity (units) of supplier $i \in I$ from "supply_capacity.csv"
- $c_{ij}$ = transportation cost per unit from supplier $i$ to customer $j$ from "transportation_costs.csv"

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   For each customer $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
2. **Supply capacity:**  
   For each supplier $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

##### Data Mapping

- $I$ (suppliers): All "supplier_id" values in "supply_capacity.csv" and "transportation_costs.csv" (source order: S1, S2, ..., S18).
- $J$ (customers): All "customer_id" values in "customer_demand.csv" and "transportation_costs.csv" (source order: C1, C2, ..., C18).
- $d_j$: For each $j \in J$, $d_j$ is the "demand_units" value from "customer_demand.csv" where "customer_id" = $j$ (table_id: file_0_view_0).
- $s_i$: For each $i \in I$, $s_i$ is the "supply_capacity_units" value from "supply_capacity.csv" where "supplier_id" = $i$ (table_id: file_1_view_0).
- $c_{ij}$: For each $i \in I$, $j \in J$, $c_{ij}$ is the value in "transportation_costs.csv" (table_id: file_2_view_0) at row "supplier_id" = $i$, column "transportation_cost_to_$j$".

All indices, parameters, and coefficients are to be taken exactly as listed in the source files, preserving source order and identifiers.