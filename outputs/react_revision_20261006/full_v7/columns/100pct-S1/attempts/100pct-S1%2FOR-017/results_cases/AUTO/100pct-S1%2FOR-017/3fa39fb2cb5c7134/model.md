##### Mathematical Model

Let $I$ be the set of suppliers (from "supply_capacity.csv"):  
$I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$

Let $J$ be the set of customer groups (from "customer_demand.csv"):  
$J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$

Let $x_{ij} \geq 0$ be the continuous quantity shipped from supplier $i \in I$ to customer $j \in J$.

Parameters:
- $d_j$: demand of customer $j$ (from "customer_demand.csv")
- $s_i$: supply capacity of supplier $i$ (from "supply_capacity.csv")
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from "transportation_costs.csv")

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction for each customer:
   \[
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   \]
2. Supply capacity for each supplier:
   \[
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

##### Data Mapping

- $I$ (suppliers): all "supplier_id" in "supply_capacity.csv" and "transportation_costs.csv" (row axis)
- $J$ (customers): all "customer_id" in "customer_demand.csv" and "transportation_costs.csv" (column axis, mapped via "transportation_cost_to_Ck" columns)
- $d_j$: "demand" column in "customer_demand.csv", indexed by "customer_id"
- $s_i$: "supply_capacity" column in "supply_capacity.csv", indexed by "supplier_id"
- $c_{ij}$: "transportation_cost_to_Ck" columns in "transportation_costs.csv", with $i$ = "supplier_id", $j$ = $k$ (e.g., "transportation_cost_to_C1" for $j$ = C1)

All indices, parameters, and coefficients are to be taken exactly as listed in the respective CSV files, preserving all identifiers and source order.