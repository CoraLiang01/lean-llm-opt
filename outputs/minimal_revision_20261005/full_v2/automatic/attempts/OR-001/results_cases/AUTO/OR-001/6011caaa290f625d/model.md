##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center $i$ to customer group $j$.

- $i \in I$ (distribution centers), $I = \{\text{S1}, \text{S2}, \ldots, \text{S18}\}$, from column "Unnamed: 0" in "supply_capacity.csv" and "transportation_costs.csv".
- $j \in J$ (customer groups), $J = \{\text{C1}, \text{C2}, \ldots, \text{C18}\}$, from column "customer" in "customer_demand.csv" and "transportation_costs.csv".

##### Parameters

- $d_j$: demand of customer group $j$, from "demand" in "customer_demand.csv".
- $s_i$: supply capacity of distribution center $i$, from "supply_capacity" in "supply_capacity.csv".
- $c_{ij}$: transportation cost per unit from $i$ to $j$, from "transportation_costs.csv" (row $i$, column $j$).

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction** (each customer group receives at least its demand):
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
2. **Supply capacity** (each distribution center does not exceed its capacity):
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
3. **Non-negativity**:
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $I$ (distribution centers): all values in column "Unnamed: 0" of "supply_capacity.csv" (table_id: file_1_view_0) and "transportation_costs.csv" (table_id: file_2_view_0).
- $J$ (customer groups): all values in column "customer" of "customer_demand.csv" (table_id: file_0_view_0) and columns "C1"–"C18" of "transportation_costs.csv" (table_id: file_2_view_0).
- $d_j$: value in column "demand" for customer $j$ in "customer_demand.csv" (table_id: file_0_view_0).
- $s_i$: value in column "supply_capacity" for distribution center $i$ in "supply_capacity.csv" (table_id: file_1_view_0).
- $c_{ij}$: value in row $i$ (identified by "Unnamed: 0") and column $j$ in "transportation_costs.csv" (table_id: file_2_view_0).

All indices, coefficients, and constraints are bound directly to the retrieved data. No data is omitted or invented.