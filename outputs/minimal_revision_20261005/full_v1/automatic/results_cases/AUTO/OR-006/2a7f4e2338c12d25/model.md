##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i$ to retail store $j$, for all warehouses $i$ and stores $j$.

##### Objective Function

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. Demand satisfaction (each store receives at least its demand):
$$
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
$$

2. Supply capacity (each warehouse ships no more than its capacity):
$$
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
$$

3. Non-negativity:
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

---

#### Data Mapping

- Sets:
    - Warehouses $I$: All values in column "Unnamed: 0" of table_id file_1_view_0 (supply_capacity.csv): S1, S2, S3, S4, S5, S6, S7, S8, S9, S10
    - Stores $J$: All values in column "customer" of table_id file_0_view_0 (customer_demand.csv): C1, C2, C3, C4, C5, C6, C7, C8, C9, C10

- Parameters:
    - Demand $d_j$: For each $j$, value in column "demand" of table_id file_0_view_0, row where "customer" = $j$
    - Supply capacity $s_i$: For each $i$, value in column "supply_capacity" of table_id file_1_view_0, row where "Unnamed: 0" = $i$
    - Transportation cost $c_{ij}$: For each $i,j$, value in table_id file_2_view_0 (transportation_costs.csv), row where "Unnamed: 0" = $i$, column $j$

- Variable:
    - $x_{ij}$: quantity shipped from warehouse $i$ to store $j$, continuous, $\geq 0$

- All indices and coefficients are to be taken exactly as listed in the respective CSV files, preserving source order.

---

##### Data Table References

- customer_demand.csv (table_id: file_0_view_0): columns ["customer", "demand"]
- supply_capacity.csv (table_id: file_1_view_0): columns ["Unnamed: 0", "supply_capacity"]
- transportation_costs.csv (table_id: file_2_view_0): matrix with rows indexed by "Unnamed: 0" (warehouses), columns by store IDs (C1, ..., C10)

---

This model uses all identifiers and coefficients as provided in the source data, with no aggregation or omission.