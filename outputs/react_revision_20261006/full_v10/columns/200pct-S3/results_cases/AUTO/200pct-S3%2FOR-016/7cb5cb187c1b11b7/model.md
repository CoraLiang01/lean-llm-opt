##### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups, as defined by the source data.

Decision variables:
For each $i\in I$, $j\in J$:
$$
x_{ij} \geq 0
$$
where $x_{ij}$ is the quantity shipped from distribution center $i$ to customer group $j$ (continuous).

Objective:
$$
\min \sum_{i\in I} \sum_{j\in J} c_{ij} x_{ij}
$$
where $c_{ij}$ is the transportation cost per unit from $i$ to $j$.

Subject to:

1. Demand satisfaction (for each customer group $j$):
$$
\sum_{i\in I} x_{ij} \geq d_j \quad \forall j \in J
$$
where $d_j$ is the demand of customer $j$.

2. Supply capacity (for each distribution center $i$):
$$
\sum_{j\in J} x_{ij} \leq s_i \quad \forall i \in I
$$
where $s_i$ is the supply capacity of supplier $i$.

3. Non-negativity:
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

##### Data Mapping

- $I$: All supplier_id in file_1_view_0 (supply_capacity.csv), source order.
- $J$: All customer_id in file_0_view_0 (customer_demand.csv), source order.
- $d_j$: demand_units for customer_id $j$ in file_0_view_0 (customer_demand.csv).
- $s_i$: supply_capacity_units for supplier_id $i$ in file_1_view_0 (supply_capacity.csv).
- $c_{ij}$: transportation_cost_to_$j$ for supplier_id $i$ in file_2_view_0 (transportation_costs.csv), using the column mapping in the Observation.

Index sets, parameters, and all coefficients are to be taken exactly as listed in the current source data, preserving all identifiers and source order.