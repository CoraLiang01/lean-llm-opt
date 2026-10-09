##### Mathematical Model

Let $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$ be the set of suppliers, and $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$ the set of customer groups.

Decision variables:
$$
x_{ij} \geq 0 \quad \text{(continuous)}, \quad \forall i \in I,\, j \in J
$$
where $x_{ij}$ is the quantity shipped from supplier $i$ to customer group $j$.

Objective:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:

1. **Demand satisfaction** (each customer group receives at least its demand):
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

- $I$ (suppliers): all "Unnamed: 0" values from table_id file_1_view_0
- $J$ (customers): all "customer" values from table_id file_0_view_0
- $d_j$: "demand" for customer $j$ from table_id file_0_view_0, column "demand"
- $s_i$: "supply_capacity" for supplier $i$ from table_id file_1_view_0, column "supply_capacity"
- $c_{ij}$: value in table_id file_2_view_0, row with "Unnamed: 0" = $i$, column $j$ (column names "C1", ..., "C10")

All indices, parameters, and coefficients are to be taken exactly as listed in the current Observation.