##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).

##### Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$ (warehouses)
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$ (stores)

##### Parameters

- $d_j$: demand of store $j \in J$ (from customer_demand.csv)
- $s_i$: supply capacity of warehouse $i \in I$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from transportation_costs.csv)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:** Each store's demand must be met:
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
2. **Supply capacity:** Each warehouse cannot ship more than its capacity:
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
3. **Non-negativity:**
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Data Mapping

- $I$ (warehouses): region column in supply_capacity.csv and Unnamed: 0 in transportation_costs.csv
- $J$ (stores): customer column in customer_demand.csv and columns D1–D5 in transportation_costs.csv
- $d_j$: demand column in customer_demand.csv, indexed by customer
- $s_i$: supply_capacity column in supply_capacity.csv, indexed by region
- $c_{ij}$: value at row $i$ (Unnamed: 0 = region) and column $j$ (D1–D5) in transportation_costs.csv

No data omitted or aggregated; all identifiers and coefficients are preserved as in the source files.