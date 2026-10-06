#### Symbolic Model

Let $I = \{S1, S2, S3, S4, S5, S6, S7, S8, S9, S10\}$ (suppliers), $J = \{C1, C2, C3, C4, C5, C6, C7, C8, C9, C10\}$ (customer groups).

Decision variables:
$$
x_{ij} \geq 0 \quad \text{(continuous)},\quad \forall i \in I,\, j \in J
$$
where $x_{ij}$ is the quantity shipped from supplier $i$ to customer group $j$.

Objective:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:
- Demand satisfaction:
$$
\sum_{i \in I} x_{ij} \geq d_j, \quad \forall j \in J
$$
- Supply capacity:
$$
\sum_{j \in J} x_{ij} \leq s_i, \quad \forall i \in I
$$
- Nonnegativity:
$$
x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
$$

#### Data Mapping

- Demand $d_j$ for each customer group $j$ from `customer_demand.csv` (table_id: file_0_view_0, columns: customer, demand):

  | customer | demand |
  |----------|--------|
  | C1       | 216    |
  | C2       | 168    |
  | C3       | 264    |
  | C4       | 216    |
  | C5       | 216    |
  | C6       | 192    |
  | C7       | 144    |
  | C8       | 168    |
  | C9       | 168    |
  | C10      | 168    |

- Supply capacity $s_i$ for each supplier $i$ from `supply_capacity.csv` (table_id: file_1_view_0, columns: Unnamed: 0, supply_capacity):

  | Unnamed: 0 | supply_capacity |
  |------------|----------------|
  | S1         | 288            |
  | S2         | 288            |
  | S3         | 264            |
  | S4         | 264            |
  | S5         | 216            |
  | S6         | 216            |
  | S7         | 168            |
  | S8         | 216            |
  | S9         | 240            |
  | S10        | 168            |

- Transportation cost $c_{ij}$ from supplier $i$ to customer group $j$ from `transportation_costs.csv` (table_id: file_2_view_0, columns: Unnamed: 0, C1, ..., C10):

  - Each row corresponds to a supplier (Unnamed: 0: S1, ..., S10), each column to a customer group (C1, ..., C10), and each entry is the cost per unit from $i$ to $j$.

All indices, parameters, and coefficients are to be used exactly as retrieved from the source tables above.