##### Mathematical Model

Let $S$ be the set of suppliers (from "supply_capacity.csv"):  
$S = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$

Let $C$ be the set of customer groups (from "customer_demand.csv"):  
$C = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$

Let $x_{ij} \geq 0$ be the continuous quantity shipped from supplier $i \in S$ to customer $j \in C$.

Parameters:
- $d_j$: demand of customer $j$ (from "customer_demand.csv")
- $s_i$: supply capacity of supplier $i$ (from "supply_capacity.csv")
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from "transportation_costs.csv")

Objective:
\[
\min \sum_{i \in S} \sum_{j \in C} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction for each customer:
\[
\sum_{i \in S} x_{ij} \geq d_j \qquad \forall j \in C
\]
2. Supply capacity for each supplier:
\[
\sum_{j \in C} x_{ij} \leq s_i \qquad \forall i \in S
\]
3. Non-negativity:
\[
x_{ij} \geq 0 \qquad \forall i \in S,\, j \in C
\]

##### Data Mapping

- $S$ (suppliers): All "supplier_id" in "supply_capacity.csv" and "transportation_costs.csv" (row axis).
- $C$ (customers): All "customer_id" in "customer_demand.csv" and "transportation_costs.csv" (column axis, mapped as "transportation_cost_to_Ck").
- $d_j$: "demand" column in "customer_demand.csv", indexed by "customer_id".
- $s_i$: "supply_capacity" column in "supply_capacity.csv", indexed by "supplier_id".
- $c_{ij}$: "transportation_cost_to_Ck" columns in "transportation_costs.csv", with $i$ as "supplier_id" and $j$ as $Ck$.

Index sets, parameters, and all coefficients are defined by the current CSV data as described above.