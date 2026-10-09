##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i$ to store $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$ (warehouses)
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$ (stores)

##### Parameters

- $d_j$: demand of store $j$ (from "customer_demand.csv")
- $s_i$: supply capacity of warehouse $i$ (from "supply_capacity.csv")
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from "transportation_costs.csv")

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:** For each store $j \in J$,
$$
\sum_{i \in I} x_{ij} \geq d_j
$$

2. **Supply capacity:** For each warehouse $i \in I$,
$$
\sum_{j \in J} x_{ij} \leq s_i
$$

3. **Non-negativity:**
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

##### Data Mapping

- $d_j$ from "customer_demand.csv", table_id: file_0_view_0, column: "demand", index: "customer"
- $s_i$ from "supply_capacity.csv", table_id: file_1_view_0, column: "supply_capacity", index: "region"
- $c_{ij}$ from "transportation_costs.csv", table_id: file_2_view_0, rows indexed by "Unnamed: 0" (warehouse $i$), columns by store $j$ ("D1", "D2", "D3", "D4", "D5")