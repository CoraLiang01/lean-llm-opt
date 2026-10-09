##### Mathematical Model

Let $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$ be the set of warehouses, and $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$ the set of retail stores.

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).

Parameters:
- $d_j$: demand of store $j \in J$ (from customer_demand.csv)
- $s_i$: supply capacity of warehouse $i \in I$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from transportation_costs.csv)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction for each store:
   \[
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   \]
2. Supply capacity for each warehouse:
   \[
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

##### Data Mapping

- $I$ (warehouses): all "Unnamed: 0" values in supply_capacity.csv and transportation_costs.csv rows: S1, S2, S3, S4, S5, S6, S7, S8, S9, S10
- $J$ (stores): all "customer" values in customer_demand.csv and all columns (except "Unnamed: 0") in transportation_costs.csv: C1, C2, C3, C4, C5, C6, C7, C8, C9, C10
- $d_j$: from customer_demand.csv, column "demand", indexed by "customer"
- $s_i$: from supply_capacity.csv, column "supply_capacity", indexed by "Unnamed: 0"
- $c_{ij}$: from transportation_costs.csv, entry in row "Unnamed: 0" = $i$, column $j$

All indices, parameters, and coefficients are to be taken exactly as listed in the source files.