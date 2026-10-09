Mathematical Model

Sets:
Let $I$ be the set of distribution centers (suppliers): $I = \{\text{supply1}, \text{supply2}, \text{supply3}, \text{supply4}, \text{supply5}, \text{supply6}, \text{supply7}, \text{supply8}\}$ (from file_1_view_0, column supplier_id).
Let $J$ be the set of customer groups: $J = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$ (from file_0_view_0, column customer_id).

Parameters:
$d_j$: demand of customer group $j \in J$ (from file_0_view_0, column demand, indexed by customer_id).
$s_i$: supply capacity of distribution center $i \in I$ (from file_1_view_0, column supply_capacity, indexed by supplier_id).
$c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$ (from file_2_view_0, column transportation_cost_to_$j$, row supplier_id $i$).

Decision Variables:
$x_{ij} \geq 0$: quantity shipped from distribution center $i \in I$ to customer group $j \in J$ (continuous).

Objective:
Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:

1. Demand satisfaction for each customer group:
$$
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
$$

2. Supply capacity for each distribution center:
$$
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
$$

3. Non-negativity:
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

Data Mapping

- $I$ (distribution centers): file_1_view_0, column supplier_id
- $J$ (customer groups): file_0_view_0, column customer_id
- $d_j$: file_0_view_0, column demand, indexed by customer_id
- $s_i$: file_1_view_0, column supply_capacity, indexed by supplier_id
- $c_{ij}$: file_2_view_0, row supplier_id $i$, column transportation_cost_to_${j}$

All indices, parameters, and coefficients are mapped exactly as in the current Observation, preserving source order and identifiers.