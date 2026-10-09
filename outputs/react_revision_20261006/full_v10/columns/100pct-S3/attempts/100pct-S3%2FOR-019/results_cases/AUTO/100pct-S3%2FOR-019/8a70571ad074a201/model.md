##### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups (demands):

$I = \{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\}$

$J = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$

Let $x_{ij} \geq 0$ be the continuous quantity shipped from supplier $i \in I$ to customer group $j \in J$.

Parameters:
- $d_j$: demand for customer group $j$ (from customer_demand.csv)
- $s_i$: supply capacity of supplier $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer group $j$ (from transportation_costs.csv)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
\[
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
\]
\[
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
\]
\[
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\]

##### Data Mapping

- $I$ (suppliers): All unique values in column "supplier_id" of file_1_view_0 (supply_capacity.csv), source order.
- $J$ (customer groups): All unique values in column "customer_id" of file_0_view_0 (customer_demand.csv), source order.
- $d_j$: Value in column "demand" for customer $j$ in file_0_view_0 (customer_demand.csv).
- $s_i$: Value in column "supply_capacity" for supplier $i$ in file_1_view_0 (supply_capacity.csv).
- $c_{ij}$: Value in column "transportation_cost_to_{j}" for supplier $i$ in file_2_view_0 (transportation_costs.csv), where $j$ matches the customer_id in file_0_view_0.

All indices, parameters, and coefficients are to be used exactly as mapped from the current source data, preserving source order and identifiers.