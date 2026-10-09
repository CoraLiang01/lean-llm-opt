##### Mathematical Model

Let $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (production plants) and $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (retail outlets).

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

2. Supply capacity (each plant ships no more than its capacity):
$$
\sum_{j \in J} x_{ij} \leq s_i, \quad \forall i \in I
$$

3. Non-negativity:
$$
x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
$$

##### Data Mapping

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ from supply_capacity.csv (file_1_view_0, column supplier_id)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ from customer_demand.csv (file_0_view_0, column customer_id)
- $d_j$ = demand for outlet $j$ from customer_demand.csv (file_0_view_0, column demand)
- $s_i$ = supply capacity for plant $i$ from supply_capacity.csv (file_1_view_0, column supply_capacity)
- $c_{ij}$ = transportation cost per unit from plant $i$ to outlet $j$ from transportation_costs.csv (file_2_view_0, row supplier_id $i$, column transportation_cost_to_$j$)

Index and parameter values are to be taken exactly as listed in the respective CSV files.