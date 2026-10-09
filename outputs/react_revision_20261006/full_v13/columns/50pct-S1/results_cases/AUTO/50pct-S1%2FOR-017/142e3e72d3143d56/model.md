Mathematical Model

Let $I$ be the set of suppliers (from file_1_view_0, column supplier_id): $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$.

Let $J$ be the set of customer groups (from file_0_view_0, column customer_id): $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$.

Decision variables:
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$
where $x_{ij}$ is the quantity shipped from supplier $i$ to customer group $j$.

Parameters:
- $d_j$: demand of customer group $j$ (from file_0_view_0, column demand)
- $s_i$: supply capacity of supplier $i$ (from file_1_view_0, column supply_capacity)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer group $j$ (from file_2_view_0, column transportation_cost_to_Ck for each $j$)

Objective:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:
1. Demand satisfaction:
$$
\sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
$$

2. Supply capacity:
$$
\sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
$$

3. Non-negativity:
$$
x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
$$

Data Mapping

- $I$ (suppliers): file_1_view_0, column supplier_id
- $J$ (customers): file_0_view_0, column customer_id
- $d_j$: file_0_view_0, columns customer_id, demand
- $s_i$: file_1_view_0, columns supplier_id, supply_capacity
- $c_{ij}$: file_2_view_0, row supplier_id $i$, column transportation_cost_to_Ck for customer $j$ (where $k$ matches customer_id in $J$)
- $x_{ij}$: decision variable for each $(i,j)$ pair in $I \times J$

All index sets, parameters, and constraints are defined directly from the current source data, preserving all identifiers and bounds.