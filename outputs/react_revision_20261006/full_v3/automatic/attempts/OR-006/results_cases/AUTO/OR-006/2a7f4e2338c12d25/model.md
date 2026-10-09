##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i$ to store $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I$: set of warehouses, from column "Unnamed: 0" in "supply_capacity.csv" and "transportation_costs.csv" rows:
  $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$
- $J$: set of stores, from column "customer" in "customer_demand.csv" and "transportation_costs.csv" columns:
  $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$

##### Parameters

- $d_j$: demand of store $j$, from "customer_demand.csv", column "demand", indexed by "customer".
- $s_i$: supply capacity of warehouse $i$, from "supply_capacity.csv", column "supply_capacity", indexed by "Unnamed: 0".
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$, from "transportation_costs.csv", entry at row $i$ ("Unnamed: 0") and column $j$.

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction** (each store receives at least its demand):
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
2. **Supply capacity** (each warehouse ships no more than its capacity):
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
3. **Non-negativity**:
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Data Mapping

- $I$ (warehouses): "Unnamed: 0" in "supply_capacity.csv" and "transportation_costs.csv" rows.
- $J$ (stores): "customer" in "customer_demand.csv" and "transportation_costs.csv" columns.
- $d_j$: "demand" in "customer_demand.csv", indexed by "customer".
- $s_i$: "supply_capacity" in "supply_capacity.csv", indexed by "Unnamed: 0".
- $c_{ij}$: "transportation_costs.csv", entry at row $i$ ("Unnamed: 0") and column $j$.

No data omitted or invented. All indices, coefficients, and constraints are mapped directly from the retrieved source data.