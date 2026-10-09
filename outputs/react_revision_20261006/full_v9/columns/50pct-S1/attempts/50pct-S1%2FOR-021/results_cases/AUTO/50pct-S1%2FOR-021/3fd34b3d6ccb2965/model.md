Mathematical Model

Let $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ be the set of production plants (suppliers), and $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ be the set of retail outlets (customers).

Decision Variables:
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$
where $x_{ij}$ is the quantity shipped from plant $i$ to outlet $j$.

Parameters:
- $d_j$: demand of outlet $j$ (from customer_demand.csv)
- $s_i$: supply capacity of plant $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from plant $i$ to outlet $j$ (from transportation_costs.csv)

Objective:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:
1. Demand satisfaction for each outlet:
$$
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
$$

2. Supply capacity for each plant:
$$
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
$$

3. Non-negativity:
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

Data Mapping

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ from supply_capacity.csv (file_1_view_0, column supplier_id)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ from customer_demand.csv (file_0_view_0, column customer_id)
- $d_j$ from customer_demand.csv (file_0_view_0, column demand, indexed by customer_id)
- $s_i$ from supply_capacity.csv (file_1_view_0, column supply_capacity, indexed by supplier_id)
- $c_{ij}$ from transportation_costs.csv (file_2_view_0, columns transportation_cost_to_C1, ..., transportation_cost_to_C4, rows indexed by supplier_id, columns mapped to customer_id as per Observation)

All indices, parameters, and mappings are defined exactly as in the current source data.