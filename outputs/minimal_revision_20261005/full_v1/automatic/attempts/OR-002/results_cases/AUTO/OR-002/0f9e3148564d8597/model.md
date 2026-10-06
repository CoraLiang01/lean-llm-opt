#### Symbolic Model

Let $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}, \text{S11}\}$ (stores) and $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}\}$ (customer groups).

Decision variables:
$$
x_{ij} \geq 0 \quad \text{(continuous)},\quad \forall i \in I,\, j \in J
$$
where $x_{ij}$ is the quantity shipped from store $i$ to customer group $j$.

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

- $d_j$: demand for customer group $j$ from table_id file_0_view_0, column "demand", row with "customer" = $j$
- $s_i$: supply capacity for store $i$ from table_id file_1_view_0, column "supply_capacity", row with "Unnamed: 0" = $i$
- $c_{ij}$: transportation cost per unit from store $i$ to customer group $j$ from table_id file_2_view_0, column $j$, row with "Unnamed: 0" = $i$

All identifiers and coefficients are to be taken exactly as in the source tables, preserving their order and names.