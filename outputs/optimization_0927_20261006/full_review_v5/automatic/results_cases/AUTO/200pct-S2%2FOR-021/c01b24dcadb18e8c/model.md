##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity of beverages shipped from plant $i$ to retail outlet $j$, for all $i \in I$ (plants) and $j \in J$ (retail outlets).

##### Sets

- $I = \{S1, S2, S3, S4\}$ (production plants)
- $J = \{C1, C2, C3, C4\}$ (retail outlets)

##### Parameters

- Demand at each retail outlet:
  - $d_{C1} = 94$
  - $d_{C2} = 39$
  - $d_{C3} = 65$
  - $d_{C4} = 435$

- Supply capacity at each plant:
  - $s_{S1} = 2531$
  - $s_{S2} = 20$
  - $s_{S3} = 210$
  - $s_{S4} = 241$

- Transportation costs per unit from each plant to each outlet:

|         | C1              | C2                | C3                | C4                |
|---------|-----------------|-------------------|-------------------|-------------------|
| S1      | 543.756480860856| 23.685276141764653| 23.676386730773032| 447.75143678673766|
| S2      | 883.9151090405642| 0.04977684765576961| 0.0350986687216299| 44.45588531711622 |
| S3      | 537.3456896658107| 23.769274659075112| 498.95659249465467| 440.60737890439776|
| S4      |1791.493192397229 | 68.21633865655126 |1432.4837339656747 |1527.7635425462734 |

Let $c_{ij}$ denote the cost from plant $i$ to outlet $j$ as above.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:** Each retail outlet must receive at least its demand.
   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]

2. **Supply capacity:** Each plant cannot ship more than its supply capacity.
   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]

3. **Non-negativity:**
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

##### Parameter Tables

**Demand:**

| customer_id | demand |
|-------------|--------|
| C1          | 94     |
| C2          | 39     |
| C3          | 65     |
| C4          | 435    |

**Supply Capacity:**

| supplier_id | supply_capacity |
|-------------|----------------|
| S1          | 2531           |
| S2          | 20             |
| S3          | 210            |
| S4          | 241            |

**Transportation Costs:**

| supplier_id | transportation_cost_to_C1 | transportation_cost_to_C2 | transportation_cost_to_C3 | transportation_cost_to_C4 |
|-------------|--------------------------|--------------------------|--------------------------|--------------------------|
| S1          | 543.756480860856         | 23.685276141764653       | 23.676386730773032       | 447.75143678673766       |
| S2          | 883.9151090405642        | 0.04977684765576961      | 0.0350986687216299       | 44.45588531711622        |
| S3          | 537.3456896658107        | 23.769274659075112       | 498.95659249465467       | 440.60737890439776       |
| S4          | 1791.493192397229        | 68.21633865655126        | 1432.4837339656747       | 1527.7635425462734       |