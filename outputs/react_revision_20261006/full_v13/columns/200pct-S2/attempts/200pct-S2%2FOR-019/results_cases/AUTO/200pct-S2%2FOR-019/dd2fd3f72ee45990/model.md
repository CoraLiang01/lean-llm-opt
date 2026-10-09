##### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups (demands):

- $I = \{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\}$
- $J = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$

Define variables:
- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to customer group $j \in J$ (continuous).

Parameters:
- $d_j$: demand of customer group $j$ (from file_0_view_0, column "demand").
- $s_i$: supply capacity of supplier $i$ (from file_1_view_0, column "supply_capacity").
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer group $j$ (from file_2_view_0, columns "transportation_cost_to_demand1" through "transportation_cost_to_demand8").

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction:
   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]
2. Supply capacity:
   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

##### Data Mapping

- $I$ (suppliers): All "supplier_id" in file_1_view_0 (supply_capacity.csv), source order.
- $J$ (customer groups): All "customer_id" in file_0_view_0 (customer_demand.csv), source order.
- $d_j$: file_0_view_0, column "demand", indexed by "customer_id".
- $s_i$: file_1_view_0, column "supply_capacity", indexed by "supplier_id".
- $c_{ij}$: file_2_view_0, row "supplier_id" (matching $i$), column "transportation_cost_to_{j}$" (matching $j$).

Index and parameter sets are defined by the full returned entity lists; all constraints and variables are as above. No data is omitted or aggregated.