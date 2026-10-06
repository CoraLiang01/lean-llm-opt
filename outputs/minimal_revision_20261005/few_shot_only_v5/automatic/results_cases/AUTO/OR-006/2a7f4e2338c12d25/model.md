##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i$ to store $j$, for all warehouses $i \in I$ and stores $j \in J$.

##### Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$ (warehouses, in source order)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$ (stores, in source order)

##### Parameters

- Demand $d_j$ for each store $j$ (from customer_demand.csv, source order):

  - $d_{\text{C1}} = 45$
  - $d_{\text{C2}} = 23$
  - $d_{\text{C3}} = 94$
  - $d_{\text{C4}} = 92$
  - $d_{\text{C5}} = 57$
  - $d_{\text{C6}} = 52$
  - $d_{\text{C7}} = 23$
  - $d_{\text{C8}} = 99$
  - $d_{\text{C9}} = 99$
  - $d_{\text{C10}} = 77$

- Supply capacity $s_i$ for each warehouse $i$ (from supply_capacity.csv, source order):

  - $s_{\text{S1}} = 127$
  - $s_{\text{S2}} = 236$
  - $s_{\text{S3}} = 168$
  - $s_{\text{S4}} = 115$
  - $s_{\text{S5}} = 280$
  - $s_{\text{S6}} = 179$
  - $s_{\text{S7}} = 135$
  - $s_{\text{S8}} = 263$
  - $s_{\text{S9}} = 283$
  - $s_{\text{S10}} = 476$

- Transportation cost per unit $c_{ij}$ from warehouse $i$ to store $j$ (from transportation_costs.csv, source order):

  - For all $i \in I$, $j \in J$, $c_{ij}$ as in the table below (rows: S1–S10, columns: C1–C10):

|      | C1           | C2           | C3           | C4           | C5           | C6           | C7           | C8           | C9           | C10          |
|------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| S1   | 2077.0586725 | 0.0          | 54.33526480  | 0.0          | 0.0          | 36.17284629  | 0.0          | 0.0          | 169.33026927 | 0.0          |
| S2   | 2077.0586725 | 0.0          | 1141.0405609 | 0.0          | 0.0          | 651.11123325 | 0.0          | 0.0          | 8.06334616   | 0.0          |
| S3   | 79.92102960  | 474.24509131 | 1477.0676289 | 22.58309959  | 474.24509131 | 41.10659696  | 474.24509131 | 474.24509131 | 624.16253950 | 474.24509131 |
| S4   | 1659.3369291 | 57.20541469  | 186.15190481 | 1201.3137084 | 1029.6974644 | 41.82210594  | 57.20541469  | 1201.3137084 | 884.56338707 | 1029.6974644 |
| S5   | 1297.2567041 | 77.76629131  | 24.26760228  | 1399.7932436 | 77.76629131  | 53.91161728  | 1399.7932436 | 77.76629131  | 1255.1151480 | 1399.7932436 |
| S6   | 1998.9090659 | 985.31654357 | 2.85416869   | 1149.5359675 | 985.31654357 | 730.69236477 | 54.73980798  | 985.31654357 | 46.80310221  | 1149.5359675 |
| S7   | 1780.3360050 | 0.0          | 1141.0405609 | 0.0          | 0.0          | 36.17284629  | 0.0          | 0.0          | 8.06334616   | 0.0          |
| S8   | 75.40935896  | 1338.1987291 | 21.39134599  | 74.34437384  | 74.34437384  | 937.35062391 | 1338.1987291 | 1338.1987291 | 1392.1186581 | 1338.1987291 |
| S9   | 98.90755583  | 0.0          | 978.03476648 | 0.0          | 0.0          | 651.11123325 | 0.0          | 0.0          | 169.33026927 | 0.0          |
| S10  | 2077.0586725 | 0.0          | 54.33526480  | 0.0          | 0.0          | 36.17284629  | 0.0          | 0.0          | 145.14023080 | 0.0          |

##### Mathematical Model

**Objective:**
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

**Subject to:**

- **Demand satisfaction (for each store $j$):**
  \[
  \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
  \]

- **Supply capacity (for each warehouse $i$):**
  \[
  \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
  \]

- **Non-negativity:**
  \[
  x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
  \]

---

#### Data Mapping

- customer_demand.csv: customer $\to$ $j$, demand $\to$ $d_j$
- supply_capacity.csv: Unnamed: 0 $\to$ $i$, supply_capacity $\to$ $s_i$
- transportation_costs.csv: Unnamed: 0 $\to$ $i$, $Ck$ columns $\to$ $c_{ij}$ for $j = Ck$