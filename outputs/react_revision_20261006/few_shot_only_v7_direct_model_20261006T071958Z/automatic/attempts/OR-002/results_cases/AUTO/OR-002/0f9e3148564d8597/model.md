##### Mathematical Model

Let $I$ be the set of suppliers (stores) and $J$ the set of customer groups, as defined by the retrieved data.

Decision variables:
$$
x_{ij} \geq 0 \quad \text{(continuous)}, \quad \forall i \in I,\, j \in J
$$
where $x_{ij}$ is the quantity shipped from supplier $i$ to customer $j$.

Objective:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:
1. **Demand satisfaction** (each customer receives at least its demand):
   $$
   \sum_{i \in I} x_{ij} \geq d_j, \quad \forall j \in J
   $$
2. **Supply capacity** (each supplier does not exceed its capacity):
   $$
   \sum_{j \in J} x_{ij} \leq s_i, \quad \forall i \in I
   $$
3. **Non-negativity**:
   $$
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   $$

##### Data Mapping

- $I$ (suppliers): all "Unnamed: 0" values from table_id file_1_view_0 (supply_capacity.csv)
- $J$ (customers): all "customer" values from table_id file_0_view_0 (customer_demand.csv)
- $d_j$: "demand" for customer $j$ from table_id file_0_view_0 (customer_demand.csv)
- $s_i$: "supply_capacity" for supplier $i$ from table_id file_1_view_0 (supply_capacity.csv)
- $c_{ij}$: value in column $j$ (customer) and row $i$ (supplier) from table_id file_2_view_0 (transportation_costs.csv), with supplier from "Unnamed: 0" and customer from column header

All indices, parameters, and coefficients are to be taken exactly as listed in the current Observation, preserving all identifiers and source order.