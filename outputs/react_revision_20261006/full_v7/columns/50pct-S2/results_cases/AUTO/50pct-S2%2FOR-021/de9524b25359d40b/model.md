##### Mathematical Model

Let $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (production plants), $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (retail outlets).

Decision variables:
$$
x_{ij} \geq 0 \quad \text{(continuous)}, \quad \forall i \in I,\, j \in J
$$
where $x_{ij}$ is the quantity shipped from plant $i$ to outlet $j$.

Objective:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:

1. Demand satisfaction (each outlet receives at least its demand):
$$
\sum_{i \in I} x_{ij} \geq d_j, \quad \forall j \in J
$$

2. Plant capacity (no plant ships more than its capacity):
$$
\sum_{j \in J} x_{ij} \leq s_i, \quad \forall i \in I
$$

3. Non-negativity:
$$
x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
$$

##### Data Mapping

- $I$ (plants): S1, S2, S3, S4 (from supply_capacity.csv, supplier_id, table_id: file_1_view_0)
- $J$ (outlets): C1, C2, C3, C4 (from customer_demand.csv, customer_id, table_id: file_0_view_0)
- $d_j$: demand for outlet $j$ (from customer_demand.csv, demand, table_id: file_0_view_0)
- $s_i$: supply capacity of plant $i$ (from supply_capacity.csv, supply_capacity, table_id: file_1_view_0)
- $c_{ij}$: transportation cost per unit from plant $i$ to outlet $j$ (from transportation_costs.csv, transportation_cost_to_C*, table_id: file_2_view_0, with supplier_id as row and C1–C4 as columns)

All indices, parameters, and coefficients are mapped directly from the current CSV files as described above.