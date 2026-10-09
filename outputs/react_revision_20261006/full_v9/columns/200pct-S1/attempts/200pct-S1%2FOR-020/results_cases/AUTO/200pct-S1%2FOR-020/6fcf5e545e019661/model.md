##### Mathematical Model

Let $I$ be the set of warehouses (indexed by $i$), and $J$ the set of stores (indexed by $j$):

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).

Parameters:
- $d_j$: demand at store $j$ (units)
- $s_i$: supply capacity at warehouse $i$ (units)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Store demand satisfaction:
   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]
2. Warehouse supply capacity:
   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

##### Data Mapping

- $I$ (warehouses): All values in column supplier_id of table_id file_1_view_0 (supply_capacity.csv)
- $J$ (stores): All values in column customer_id of table_id file_0_view_0 (customer_demand.csv)
- $d_j$: demand_units from table_id file_0_view_0, indexed by customer_id
- $s_i$: supply_capacity_units from table_id file_1_view_0, indexed by supplier_id
- $c_{ij}$: transportation_cost_to_Dk from table_id file_2_view_0, where row supplier_id = $i$, column = transportation_cost_to_Dk for $j = \text{D}k$

Variable domain and all constraints are as specified in the user query and mapped to the current data. No data omitted or invented.