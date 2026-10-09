##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from supplier $i$ to customer $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I$: set of suppliers, $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$
- $J$: set of customers, $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$

##### Parameters

- $d_j$: demand of customer $j$, from "customer_demand.csv", table_id: file_0_view_0, column: demand
- $s_i$: supply capacity of supplier $i$, from "supply_capacity.csv", table_id: file_1_view_0, column: supply_capacity
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$, from "transportation_costs.csv", table_id: file_2_view_0, row: $i$ (column "Unnamed: 0"), column: $j$

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction** (each customer receives at least its demand):
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
2. **Supply capacity** (each supplier ships no more than its capacity):
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
3. **Non-negativity**:
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Data Mapping

- $I$ (suppliers): all "Unnamed: 0" values in "supply_capacity.csv" (table_id: file_1_view_0)
- $J$ (customers): all "customer" values in "customer_demand.csv" (table_id: file_0_view_0)
- $d_j$: "demand" column in "customer_demand.csv" (table_id: file_0_view_0), indexed by "customer"
- $s_i$: "supply_capacity" column in "supply_capacity.csv" (table_id: file_1_view_0), indexed by "Unnamed: 0"
- $c_{ij}$: value at row $i$ ("Unnamed: 0") and column $j$ in "transportation_costs.csv" (table_id: file_2_view_0)

No data is omitted or aggregated; all identifiers and coefficients are preserved as in the source.