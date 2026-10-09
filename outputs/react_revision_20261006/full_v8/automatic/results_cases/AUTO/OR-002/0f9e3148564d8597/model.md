##### Mathematical Model

Let $I$ be the set of stores (indexed by $i$), and $J$ the set of customer groups (indexed by $j$):

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}, \text{S11}\}$
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}\}$

**Decision Variables:**

For each $i \in I$, $j \in J$:
- $x_{ij} \geq 0$: quantity shipped from store $i$ to customer group $j$ (continuous).

**Parameters:**

- $d_j$: demand of customer group $j$ (from customer_demand.csv)
- $s_i$: supply capacity of store $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from store $i$ to customer group $j$ (from transportation_costs.csv)

**Objective:**

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

**Subject to:**

1. **Demand satisfaction:**  
   For all $j \in J$,
   \[
   \sum_{i \in I} x_{ij} \geq d_j
   \]

2. **Supply capacity:**  
   For all $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq s_i
   \]

3. **Non-negativity:**  
   For all $i \in I$, $j \in J$,
   \[
   x_{ij} \geq 0
   \]

##### Data Mapping

- $I$ (stores): All values in column "Unnamed: 0" of table_id file_1_view_0 (supply_capacity.csv) and row IDs of table_id file_2_view_0 (transportation_costs.csv).
- $J$ (customer groups): All values in column "customer" of table_id file_0_view_0 (customer_demand.csv) and columns of table_id file_2_view_0 (transportation_costs.csv) except "Unnamed: 0".
- $d_j$: For each $j$, value in column "demand" of table_id file_0_view_0 where "customer" = $j$.
- $s_i$: For each $i$, value in column "supply_capacity" of table_id file_1_view_0 where "Unnamed: 0" = $i$.
- $c_{ij}$: For each $i$, $j$, value in column $j$ of table_id file_2_view_0 where "Unnamed: 0" = $i$.

All index sets, parameters, and constraints are defined exactly as in the current CSV data, preserving all identifiers and their source order.