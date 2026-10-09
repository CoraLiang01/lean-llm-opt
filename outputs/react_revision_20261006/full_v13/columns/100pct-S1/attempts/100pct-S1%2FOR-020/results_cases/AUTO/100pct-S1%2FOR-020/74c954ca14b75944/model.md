##### Mathematical Model

Let $I$ be the set of warehouses (indexed by $i$), and $J$ the set of stores (indexed by $j$):

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$

Parameters:
- $d_j$: daily demand at store $j$ (from customer_demand.csv)
- $s_i$: daily supply capacity at warehouse $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from transportation_costs.csv)

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from warehouse $i$ to store $j$ (continuous)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction (each store receives at least its demand):
   \[
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   \]
2. Supply capacity (each warehouse ships no more than its capacity):
   \[
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

##### Data Mapping

- $I$ (warehouses): all supplier_id in file_1_view_0 (supply_capacity.csv), source order.
- $J$ (stores): all customer_id in file_0_view_0 (customer_demand.csv), source order.
- $d_j$: demand_units for customer_id $j$ in file_0_view_0 (customer_demand.csv).
- $s_i$: supply_capacity_units for supplier_id $i$ in file_1_view_0 (supply_capacity.csv).
- $c_{ij}$: transportation_cost_to_$j$ for supplier_id $i$ in file_2_view_0 (transportation_costs.csv), with $j$ mapped as in the relationships.column_id_mapping.
- $x_{ij}$: decision variable for shipment from $i$ to $j$, defined for all $i \in I$, $j \in J$.

All index sets, parameters, and constraints are defined exactly as in the current source data, preserving all identifiers and source order.