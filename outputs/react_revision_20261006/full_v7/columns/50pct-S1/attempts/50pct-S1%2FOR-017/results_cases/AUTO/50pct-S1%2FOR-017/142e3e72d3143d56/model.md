##### Mathematical Model

Let $S$ be the set of suppliers (from "supply_capacity.csv"):  
$S = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$

Let $C$ be the set of customer groups (from "customer_demand.csv"):  
$C = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$

Let $x_{ij} \geq 0$ be the continuous quantity shipped from supplier $i \in S$ to customer $j \in C$.

Parameters:
- $d_j$: demand of customer $j \in C$ (from "customer_demand.csv")
- $s_i$: supply capacity of supplier $i \in S$ (from "supply_capacity.csv")
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

- $S$ (suppliers): all "supplier_id" in table_id file_1_view_0 (supply_capacity.csv)
- $C$ (customers): all "customer_id" in table_id file_0_view_0 (customer_demand.csv)
- $d_j$: "demand" for customer $j$ in file_0_view_0 (customer_demand.csv)
- $s_i$: "supply_capacity" for supplier $i$ in file_1_view_0 (supply_capacity.csv)
- $c_{ij}$: "transportation_cost_to_$j$" for supplier $i$ in file_2_view_0 (transportation_costs.csv), with $j$ matched to customer_id in file_0_view_0

- Decision variables $x_{ij}$: continuous, nonnegative, for all $i \in S$, $j \in C$.

- All index sets, parameters, and coefficients are defined by the full, unsimplified, and unfiltered contents of the current CSV files as described above.