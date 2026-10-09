##### Mathematical Model

Let $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (production plants) and $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (retail outlets).

Let $x_{ij} \geq 0$ be the quantity shipped from plant $i \in I$ to outlet $j \in J$ (continuous).

Parameters:
- $d_j$: demand at outlet $j$ (from customer_demand.csv)
- $s_i$: supply capacity at plant $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from plant $i$ to outlet $j$ (from transportation_costs.csv)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction (each outlet receives at least its demand):
   \[
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   \]
2. Supply capacity (each plant ships no more than its capacity):
   \[
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

##### Data Mapping

- $I$ (plants): supplier_id in supply_capacity.csv and transportation_costs.csv: S1, S2, S3, S4
- $J$ (outlets): customer_id in customer_demand.csv and transportation_costs.csv: C1, C2, C3, C4
- $d_j$: demand column in customer_demand.csv, indexed by customer_id
- $s_i$: supply_capacity column in supply_capacity.csv, indexed by supplier_id
- $c_{ij}$: transportation_cost_to_Ck columns in transportation_costs.csv, with supplier_id as row and Ck as column (e.g., $c_{\text{S1},\text{C1}}$ from row S1, column transportation_cost_to_C1)

All indices, parameters, and coefficients are to be taken exactly as listed in the respective CSV files.