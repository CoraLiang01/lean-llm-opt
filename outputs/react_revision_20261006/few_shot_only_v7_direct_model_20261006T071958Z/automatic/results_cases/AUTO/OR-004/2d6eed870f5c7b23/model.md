##### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups, as defined by the current data.

Decision variables:
$$
x_{ij} \geq 0 \quad \text{(continuous)}, \quad \forall i \in I,\, j \in J
$$
where $x_{ij}$ is the quantity shipped from distribution center $i$ to customer group $j$.

Objective:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:
1. **Demand satisfaction** (each customer group receives at least its demand):
   $$
   \sum_{i \in I} x_{ij} \geq d_j, \quad \forall j \in J
   $$
2. **Supply capacity** (each distribution center does not exceed its capacity):
   $$
   \sum_{j \in J} x_{ij} \leq s_i, \quad \forall i \in I
   $$
3. **Non-negativity**:
   $$
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   $$

##### Data Mapping

- $I$ (distribution centers): all "Unnamed: 0" values from table_id="file_1_view_0" (supply_capacity.csv)
- $J$ (customer groups): all "customer" values from table_id="file_0_view_0" (customer_demand.csv)
- $d_j$: "demand" value for customer $j$ from table_id="file_0_view_0" (customer_demand.csv)
- $s_i$: "supply_capacity" value for supplier $i$ from table_id="file_1_view_0" (supply_capacity.csv)
- $c_{ij}$: value in row with "Unnamed: 0" = $i$ and column $j$ in table_id="file_2_view_0" (transportation_costs.csv)
- $x_{ij}$: decision variable for quantity shipped from $i$ to $j$ (continuous, $\geq 0$)

All index sets, parameters, and coefficients are to be taken exactly as listed in the current Observation, preserving all identifiers and source order.