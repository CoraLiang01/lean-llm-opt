#### Mathematical Model

Let:
- $I$ = set of production plants = $\{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (from supply_capacity.csv, file_1_view_0, supplier_id)
- $J$ = set of retail outlets = $\{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (from customer_demand.csv, file_0_view_0, customer_id)

Parameters:
- $d_j$ = demand at outlet $j \in J$ (from file_0_view_0, demand)
- $s_i$ = supply capacity at plant $i \in I$ (from file_1_view_0, supply_capacity)
- $c_{ij}$ = transportation cost per unit from plant $i$ to outlet $j$ (from file_2_view_0, transportation_cost_to_Ck)

Decision variables:
- $x_{ij} \geq 0$ = quantity shipped from plant $i$ to outlet $j$ (continuous)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction at each outlet:
   \[
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   \]
2. Supply capacity at each plant:
   \[
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

#### Data Mapping

- $I$ (plants): file_1_view_0, column supplier_id
- $J$ (outlets): file_0_view_0, column customer_id
- $d_j$: file_0_view_0, column demand, indexed by customer_id
- $s_i$: file_1_view_0, column supply_capacity, indexed by supplier_id
- $c_{ij}$: file_2_view_0, columns transportation_cost_to_C1, ..., transportation_cost_to_C4, rows indexed by supplier_id, columns mapped to customer_id as per relationships in the Observation

All indices, parameters, and constraints are defined directly from the current source data. No data is omitted or aggregated. Variable domains and all bounds are as specified in the user query and source data.