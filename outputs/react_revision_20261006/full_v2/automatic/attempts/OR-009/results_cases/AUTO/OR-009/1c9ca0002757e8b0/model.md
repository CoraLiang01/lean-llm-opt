##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from plant $i$ to retail outlet $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (plants, from supply_capacity.csv and transportation_costs.csv row IDs)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (retail outlets, from customer_demand.csv and transportation_costs.csv column IDs)

##### Parameters

- $d_j$: demand at outlet $j$ (from customer_demand.csv, column "demand", indexed by "customer")
- $s_i$: supply capacity at plant $i$ (from supply_capacity.csv, column "supply_capacity", indexed by "Unnamed: 0")
- $c_{ij}$: transportation cost per unit from plant $i$ to outlet $j$ (from transportation_costs.csv, entry at row $i$ and column $j$)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:** For each outlet $j \in J$,
$$
\sum_{i \in I} x_{ij} \geq d_j
$$

2. **Supply capacity:** For each plant $i \in I$,
$$
\sum_{j \in J} x_{ij} \leq s_i
$$

3. **Non-negativity:**
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

##### Data Mapping

- $I$ (plants): all "Unnamed: 0" values in supply_capacity.csv and transportation_costs.csv rows
- $J$ (retail outlets): all "customer" values in customer_demand.csv and all columns (except "Unnamed: 0") in transportation_costs.csv
- $d_j$: from customer_demand.csv, column "demand", indexed by "customer"
- $s_i$: from supply_capacity.csv, column "supply_capacity", indexed by "Unnamed: 0"
- $c_{ij}$: from transportation_costs.csv, entry at row "Unnamed: 0" = $i$, column $j$ (where $j$ matches "C1", "C2", "C3", "C4")

All indices, parameters, and coefficients are to be taken exactly as listed in the source files and columns.