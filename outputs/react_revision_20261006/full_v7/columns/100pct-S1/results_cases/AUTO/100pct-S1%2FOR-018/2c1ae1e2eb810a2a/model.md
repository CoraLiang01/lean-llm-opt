##### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups:
- $I = \{\text{S1}, \text{S2}, \ldots, \text{S12}\}$
- $J = \{\text{C1}, \text{C2}, \ldots, \text{C12}\}$

Let $x_{ij} \geq 0$ be the continuous quantity shipped from supplier $i \in I$ to customer $j \in J$.

Parameters:
- $d_j$: demand of customer $j$ (from customer_demand.csv)
- $s_i$: supply capacity of supplier $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from transportation_costs.csv)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction:
   \[
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   \]
2. Supply capacity:
   \[
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

##### Data Mapping

- $I$ (suppliers): All unique values in column "supplier_id" of supply_capacity.csv and transportation_costs.csv.
- $J$ (customers): All unique values in column "customer_id" of customer_demand.csv and as suffixes in transportation_costs.csv columns.
- $d_j$: "demand" column in customer_demand.csv, indexed by "customer_id".
- $s_i$: "supply_capacity" column in supply_capacity.csv, indexed by "supplier_id".
- $c_{ij}$: "transportation_cost_to_Ck" columns in transportation_costs.csv, where $i$ is "supplier_id" and $j$ is the customer suffix in the column name.

Index sets, parameters, and all coefficients are defined exactly as in the retrieved data. No data is omitted or aggregated. Variable domains and all constraints are as specified in the user query.