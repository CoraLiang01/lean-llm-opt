##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i$ to store $j$, for all warehouses $i$ and stores $j$.

##### Sets

- Warehouses $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$ (from column "Unnamed: 0" in table_id file_1_view_0)
- Stores $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$ (from column "customer" in table_id file_0_view_0)

##### Parameters

- Demand for each store $j$: $d_j$ (from column "demand" in table_id file_0_view_0)
- Supply capacity for each warehouse $i$: $s_i$ (from column "supply_capacity" in table_id file_1_view_0)
- Transportation cost per unit from warehouse $i$ to store $j$: $c_{ij}$ (from table_id file_2_view_0, row "Unnamed: 0" = $i$, column $j$)

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

3. **Non-negativity:** For all $i \in I$, $j \in J$,
$$
x_{ij} \geq 0
$$

##### Data Mapping

- $I$ (warehouses): all values in column "Unnamed: 0" of table_id file_1_view_0
- $J$ (stores): all values in column "customer" of table_id file_0_view_0
- $d_j$: value in column "demand" for store $j$ in table_id file_0_view_0
- $s_i$: value in column "supply_capacity" for warehouse $i$ in table_id file_1_view_0
- $c_{ij}$: value in table_id file_2_view_0, row "Unnamed: 0" = $i$, column $j$