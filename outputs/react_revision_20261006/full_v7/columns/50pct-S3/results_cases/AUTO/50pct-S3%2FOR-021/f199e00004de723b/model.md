##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity of beverages shipped from plant $i$ to retail outlet $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (production plants)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (retail outlets)

##### Parameters

- $d_j$: demand at outlet $j$ (from customer_demand.csv)
- $s_i$: supply capacity at plant $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from plant $i$ to outlet $j$ (from transportation_costs.csv)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. Demand satisfaction at each outlet:
$$
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
$$

2. Supply capacity at each plant:
$$
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
$$

3. Non-negativity:
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

##### Data Mapping

- $I$ (plants): supplier_id in supply_capacity.csv and transportation_costs.csv: S1, S2, S3, S4
- $J$ (outlets): customer_id in customer_demand.csv and transportation_costs.csv: C1, C2, C3, C4
- $d_j$: demand column in customer_demand.csv, indexed by customer_id
- $s_i$: supply_capacity column in supply_capacity.csv, indexed by supplier_id
- $c_{ij}$: transportation_cost_to_Ck columns in transportation_costs.csv, with supplier_id as row and Ck as column (e.g., $c_{\text{S1},\text{C1}}$ = 543.756480860856)

All indices, parameters, and coefficients are to be taken directly from the corresponding CSV files as described above.