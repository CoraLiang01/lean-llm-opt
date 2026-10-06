#### Symbolic Model

Let $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (production plants), $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (retail outlets).

Decision variables:
$$
x_{ij} \geq 0 \quad \text{(continuous)},\quad \forall i \in I,\, j \in J
$$
where $x_{ij}$ is the quantity shipped from plant $i$ to outlet $j$.

Objective:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:
1. Demand satisfaction (each outlet receives at least its demand):
   $$
   \sum_{i \in I} x_{ij} \geq d_j, \quad \forall j \in J
   $$
2. Supply capacity (each plant ships no more than its capacity):
   $$
   \sum_{j \in J} x_{ij} \leq s_i, \quad \forall i \in I
   $$
3. Non-negativity:
   $$
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   $$

#### Data Mapping

- $i$ (plants): values from column "Unnamed: 0" in table_id file_1_view_0 and file_2_view_0, in source order: S1, S2, S3, S4
- $j$ (outlets): values from column "customer" in table_id file_0_view_0 and columns C1, C2, C3, C4 in file_2_view_0, in source order: C1, C2, C3, C4

- $d_j$ (demand for outlet $j$): from table_id file_0_view_0, column "demand"
    - C1: 94
    - C2: 39
    - C3: 65
    - C4: 435

- $s_i$ (supply capacity for plant $i$): from table_id file_1_view_0, column "supply_capacity"
    - S1: 2531
    - S2: 20
    - S3: 210
    - S4: 241

- $c_{ij}$ (transportation cost per unit from plant $i$ to outlet $j$): from table_id file_2_view_0, columns C1–C4, rows S1–S4, in source order.

  | $i$ | $j$ | $c_{ij}$ |
  |-----|-----|----------|
  | S1  | C1  | 543.756480860856 |
  | S1  | C2  | 23.685276141764653 |
  | S1  | C3  | 23.676386730773032 |
  | S1  | C4  | 447.75143678673766 |
  | S2  | C1  | 883.9151090405642 |
  | S2  | C2  | 0.04977684765576961 |
  | S2  | C3  | 0.0350986687216299 |
  | S2  | C4  | 44.45588531711622 |
  | S3  | C1  | 537.3456896658107 |
  | S3  | C2  | 23.769274659075112 |
  | S3  | C3  | 498.95659249465467 |
  | S3  | C4  | 440.60737890439776 |
  | S4  | C1  | 1791.493192397229 |
  | S4  | C2  | 68.21633865655126 |
  | S4  | C3  | 1432.4837339656747 |
  | S4  | C4  | 1527.7635425462734 |

All data is mapped directly from the retrieved tables, preserving source order and identifiers.