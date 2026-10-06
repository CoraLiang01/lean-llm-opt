##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from warehouse $i$ to retail store $j$.

- $i \in I$ (warehouses), $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$
- $j \in J$ (retail stores), $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

where $c_{ij}$ is the unit transportation cost from warehouse $i$ to store $j$.

##### Constraints

1. **Demand satisfaction (for each store):**
   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]
   where $d_j$ is the demand of store $j$.

2. **Supply capacity (for each warehouse):**
   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]
   where $s_i$ is the supply capacity of warehouse $i$.

3. **Non-negativity:**
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

---

#### Data Mapping

- **Warehouses ($I$) and their supply capacities ($s_i$):**

  | table_id: file_1_view_0, column: Unnamed: 0 (warehouse), column: supply_capacity |
  |---|---|
  | S1 | 127 |
  | S2 | 236 |
  | S3 | 168 |
  | S4 | 115 |
  | S5 | 280 |
  | S6 | 179 |
  | S7 | 135 |
  | S8 | 263 |
  | S9 | 283 |
  | S10 | 476 |

- **Retail stores ($J$) and their demands ($d_j$):**

  | table_id: file_0_view_0, column: customer (store), column: demand |
  |---|---|
  | C1 | 45 |
  | C2 | 23 |
  | C3 | 94 |
  | C4 | 92 |
  | C5 | 57 |
  | C6 | 52 |
  | C7 | 23 |
  | C8 | 99 |
  | C9 | 99 |
  | C10 | 77 |

- **Transportation costs ($c_{ij}$):**

  | table_id: file_2_view_0, row: Unnamed: 0 (warehouse), columns: C1–C10 (stores) |
  |---|---|
  | $c_{ij}$ is the value in row $i$ (warehouse) and column $j$ (store) of file_2_view_0 |

---

**All parameters and indices are bound directly to the retrieved data.**