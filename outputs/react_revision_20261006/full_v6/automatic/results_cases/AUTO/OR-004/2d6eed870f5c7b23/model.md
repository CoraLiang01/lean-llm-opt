##### Mathematical Model

Let $I$ be the set of distribution centers (indexed by $i$), and $J$ the set of customer groups (indexed by $j$):

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}, \text{S11}, \text{S12}\}$
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}\}$

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center $i$ to customer group $j$.

Parameters:
- $d_j$: demand of customer group $j$ (from "customer_demand.csv")
- $s_i$: supply capacity of distribution center $i$ (from "supply_capacity.csv")
- $c_{ij}$: transportation cost per unit from $i$ to $j$ (from "transportation_costs.csv")

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

- $I$ (distribution centers): all "Unnamed: 0" values in "supply_capacity.csv" and row labels in "transportation_costs.csv"
- $J$ (customer groups): all "customer" values in "customer_demand.csv" and column labels in "transportation_costs.csv"
- $d_j$: "demand" column in "customer_demand.csv", indexed by "customer"
- $s_i$: "supply_capacity" column in "supply_capacity.csv", indexed by "Unnamed: 0"
- $c_{ij}$: entry in "transportation_costs.csv" at row $i$ ("Unnamed: 0") and column $j$ (customer group label)

Variable domain: $x_{ij} \geq 0$, continuous, for all $i \in I$, $j \in J$.