#### Mathematical Model

Let $I$ be the set of stores (from supply_capacity.csv): $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}, \text{S11}\}$.

Let $J$ be the set of customer groups (from customer_demand.csv): $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}\}$.

Let $x_{ij} \geq 0$ be the continuous quantity shipped from store $i \in I$ to customer group $j \in J$.

Parameters:
- $d_j$: demand of customer group $j$ (from customer_demand.csv)
- $s_i$: supply capacity of store $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from store $i$ to customer group $j$ (from transportation_costs.csv)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
- Demand satisfaction:
\[
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
\]
- Supply capacity:
\[
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
\]
- Non-negativity:
\[
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\]

#### Data Mapping

- $I$ (stores): all values in column "Unnamed: 0" of file_1_view_0 (supply_capacity.csv)
- $J$ (customer groups): all values in column "customer" of file_0_view_0 (customer_demand.csv)
- $d_j$: column "demand" in file_0_view_0, indexed by "customer"
- $s_i$: column "supply_capacity" in file_1_view_0, indexed by "Unnamed: 0"
- $c_{ij}$: value in file_2_view_0 (transportation_costs.csv), row "Unnamed: 0" = $i$, column $j$

- $x_{ij}$: decision variable for each $i \in I$, $j \in J$ (continuous, $\geq 0$)

All sets, parameters, and indices are defined exactly as in the current CSV data. No data is omitted or aggregated.