Let $x_{ij}$ be the number of units shipped from warehouse $i$ (with $i \in \{\text{S1}, \text{S2}, \ldots, \text{S10}\}$) to customer $j$ (with $j \in \{\text{C1}, \text{C2}, \ldots, \text{C10}\}$).

Minimize total transportation cost:
$$
\min \sum_{i \in \{\text{S1},\ldots,\text{S10}\}} \sum_{j \in \{\text{C1},\ldots,\text{C10}\}} c_{ij} x_{ij}
$$

where $c_{ij}$ is the cost per unit from warehouse $i$ to customer $j$ as given below.

Subject to:

**1. Demand satisfaction for each customer:**
$$
\sum_{i \in \{\text{S1},\ldots,\text{S10}\}} x_{ij} = d_j \qquad \forall j \in \{\text{C1},\ldots,\text{C10}\}
$$

where $d_j$ is the demand for customer $j$.

**2. Supply capacity for each warehouse:**
$$
\sum_{j \in \{\text{C1},\ldots,\text{C10}\}} x_{ij} \leq s_i \qquad \forall i \in \{\text{S1},\ldots,\text{S10}\}
$$

where $s_i$ is the supply capacity of warehouse $i$.

**3. Nonnegativity:**
$$
x_{ij} \geq 0 \qquad \forall i, j
$$

---

### Data

#### Customer Demands (from customer_demand.csv)
| customer | demand |
|----------|--------|
| C1       | 45     |
| C2       | 23     |
| C3       | 94     |
| C4       | 92     |
| C5       | 57     |
| C6       | 52     |
| C7       | 23     |
| C8       | 99     |
| C9       | 99     |
| C10      | 77     |

#### Warehouse Supply Capacities (from supply_capacity.csv)
| warehouse | supply_capacity |
|-----------|----------------|
| S1        | 127            |
| S2        | 236            |
| S3        | 168            |
| S4        | 115            |
| S5        | 280            |
| S6        | 179            |
| S7        | 135            |
| S8        | 263            |
| S9        | 283            |
| S10       | 476            |

#### Transportation Costs $c_{ij}$ (from transportation_costs.csv)

|        | C1           | C2           | C3           | C4           | C5           | C6           | C7           | C8           | C9           | C10          |
|--------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| S1     | 2077.0586725 | 0.0          | 54.33526480  | 0.0          | 0.0          | 36.17284629  | 0.0          | 0.0          | 169.3302693  | 0.0          |
| S2     | 2077.0586725 | 0.0          | 1141.0405609 | 0.0          | 0.0          | 651.1112332  | 0.0          | 0.0          | 8.06334616   | 0.0          |
| S3     | 79.92102960  | 474.2450913  | 1477.0676289 | 22.58309959  | 474.2450913  | 41.10659696  | 474.2450913  | 474.2450913  | 624.1625395  | 474.2450913  |
| S4     | 1659.3369291 | 57.20541469  | 186.1519048  | 1201.3137084 | 1029.6974644 | 41.82210594  | 57.20541469  | 1201.3137084 | 884.5633871  | 1029.6974644 |
| S5     | 1297.2567041 | 77.76629131  | 24.26760228  | 1399.7932436 | 77.76629131  | 53.91161728  | 1399.7932436 | 77.76629131  | 1255.1151480 | 1399.7932436 |
| S6     | 1998.9090659 | 985.3165436  | 2.854168689  | 1149.5359675 | 985.3165436  | 730.6923648  | 54.73980798  | 985.3165436  | 46.80310221  | 1149.5359675 |
| S7     | 1780.3360050 | 0.0          | 1141.0405609 | 0.0          | 0.0          | 36.17284629  | 0.0          | 0.0          | 8.06334616   | 0.0          |
| S8     | 75.40935896  | 1338.1987291 | 21.39134599  | 74.34437384  | 74.34437384  | 937.3506239  | 1338.1987291 | 1338.1987291 | 1392.1186581 | 1338.1987291 |
| S9     | 98.90755583  | 0.0          | 978.0347665  | 0.0          | 0.0          | 651.1112332  | 0.0          | 0.0          | 169.3302693  | 0.0          |
| S10    | 2077.0586725 | 0.0          | 54.33526480  | 0.0          | 0.0          | 36.17284629  | 0.0          | 0.0          | 145.1402308  | 0.0          |

**Variable domains:** $x_{ij} \geq 0$ and continuous (since the question does not require integer shipments).

---

**Summary of Model:**

Minimize
$$
\sum_{i=\text{S1}}^{\text{S10}} \sum_{j=\text{C1}}^{\text{C10}} c_{ij} x_{ij}
$$

Subject to
$$
\sum_{i=\text{S1}}^{\text{S10}} x_{ij} = d_j \quad \forall j \in \{\text{C1},\ldots,\text{C10}\}
$$
$$
\sum_{j=\text{C1}}^{\text{C10}} x_{ij} \leq s_i \quad \forall i \in \{\text{S1},\ldots,\text{S10}\}
$$
$$
x_{ij} \geq 0 \quad \forall i, j
$$

with all coefficients and identifiers as above.