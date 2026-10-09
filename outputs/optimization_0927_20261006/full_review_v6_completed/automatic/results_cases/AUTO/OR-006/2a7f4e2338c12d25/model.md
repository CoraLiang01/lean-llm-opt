##### Sets

- Warehouses (suppliers): $I = \{S1, S2, S3, S4, S5, S6, S7, S8, S9, S10\}$
- Retail stores (customers): $J = \{C1, C2, C3, C4, C5, C6, C7, C8, C9, C10\}$

##### Parameters

- Demand for each store $j$ ($d_j$):

  - $d_{C1} = 45$
  - $d_{C2} = 23$
  - $d_{C3} = 94$
  - $d_{C4} = 92$
  - $d_{C5} = 57$
  - $d_{C6} = 52$
  - $d_{C7} = 23$
  - $d_{C8} = 99$
  - $d_{C9} = 99$
  - $d_{C10} = 77$

- Supply capacity for each warehouse $i$ ($s_i$):

  - $s_{S1} = 127$
  - $s_{S2} = 236$
  - $s_{S3} = 168$
  - $s_{S4} = 115$
  - $s_{S5} = 280$
  - $s_{S6} = 179$
  - $s_{S7} = 135$
  - $s_{S8} = 263$
  - $s_{S9} = 283$
  - $s_{S10} = 476$

- Transportation cost per unit from warehouse $i$ to store $j$ ($c_{ij}$):

|        |   C1         |   C2         |   C3         |   C4         |   C5         |   C6         |   C7         |   C8         |   C9         |   C10        |
|--------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| S1     | 2077.0586725 | 0.0          | 54.33526480  | 0.0          | 0.0          | 36.17284629  | 0.0          | 0.0          | 169.3302693  | 0.0          |
| S2     | 2077.0586725 | 0.0          | 1141.0405609 | 0.0          | 0.0          | 651.1112332  | 0.0          | 0.0          | 8.063346156  | 0.0          |
| S3     | 79.92102960  | 474.2450913  | 1477.0676289 | 22.58309959  | 474.2450913  | 41.10659696  | 474.2450913  | 474.2450913  | 624.1625395  | 474.2450913  |
| S4     | 1659.3369291 | 57.20541469  | 186.1519048  | 1201.3137084 | 1029.6974644 | 41.82210594  | 57.20541469  | 1201.3137084 | 884.5633871  | 1029.6974644 |
| S5     | 1297.2567041 | 77.76629131  | 24.26760228  | 1399.7932436 | 77.76629131  | 53.91161728  | 1399.7932436 | 77.76629131  | 1255.1151480 | 1399.7932436 |
| S6     | 1998.9090659 | 985.3165436  | 2.854168689  | 1149.5359675 | 985.3165436  | 730.6923648  | 54.73980798  | 985.3165436  | 46.80310221  | 1149.5359675 |
| S7     | 1780.3360050 | 0.0          | 1141.0405609 | 0.0          | 0.0          | 36.17284629  | 0.0          | 0.0          | 8.063346156  | 0.0          |
| S8     | 75.40935896  | 1338.1987291 | 21.39134599  | 74.34437384  | 74.34437384  | 937.3506239  | 1338.1987291 | 1338.1987291 | 1392.1186581 | 1338.1987291 |
| S9     | 98.90755583  | 0.0          | 978.0347665  | 0.0          | 0.0          | 651.1112332  | 0.0          | 0.0          | 169.3302693  | 0.0          |
| S10    | 2077.0586725 | 0.0          | 54.33526480  | 0.0          | 0.0          | 36.17284629  | 0.0          | 0.0          | 145.1402308  | 0.0          |

##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from warehouse $i$ to store $j$ (continuous).

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
   - $j = C1: \sum_{i} x_{i,C1} \geq 45$
   - $j = C2: \sum_{i} x_{i,C2} \geq 23$
   - $j = C3: \sum_{i} x_{i,C3} \geq 94$
   - $j = C4: \sum_{i} x_{i,C4} \geq 92$
   - $j = C5: \sum_{i} x_{i,C5} \geq 57$
   - $j = C6: \sum_{i} x_{i,C6} \geq 52$
   - $j = C7: \sum_{i} x_{i,C7} \geq 23$
   - $j = C8: \sum_{i} x_{i,C8} \geq 99$
   - $j = C9: \sum_{i} x_{i,C9} \geq 99$
   - $j = C10: \sum_{i} x_{i,C10} \geq 77$

2. **Supply capacity:** For each warehouse $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
   - $i = S1: \sum_{j} x_{S1,j} \leq 127$
   - $i = S2: \sum_{j} x_{S2,j} \leq 236$
   - $i = S3: \sum_{j} x_{S3,j} \leq 168$
   - $i = S4: \sum_{j} x_{S4,j} \leq 115$
   - $i = S5: \sum_{j} x_{S5,j} \leq 280$
   - $i = S6: \sum_{j} x_{S6,j} \leq 179$
   - $i = S7: \sum_{j} x_{S7,j} \leq 135$
   - $i = S8: \sum_{j} x_{S8,j} \leq 263$
   - $i = S9: \sum_{j} x_{S9,j} \leq 283$
   - $i = S10: \sum_{j} x_{S10,j} \leq 476$

3. **Non-negativity:**
   $$
   x_{ij} \geq 0 \quad \forall i \in I, j \in J
   $$

##### Complete Model

Minimize
$$
\sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to
$$
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
$$
$$
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
$$
$$
x_{ij} \geq 0 \quad \forall i \in I, j \in J
$$

Where all parameters and indices are as listed above.