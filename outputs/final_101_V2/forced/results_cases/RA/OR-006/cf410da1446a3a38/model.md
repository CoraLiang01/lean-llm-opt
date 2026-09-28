Let $x_{ij}$ denote the quantity of product shipped from warehouse $i$ to customer $j$.

#### Sets and Indices

- Warehouses $i \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$
- Customers $j \in \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$

#### Parameters

- $d_j$: demand for customer $j$ (from customer_demand.csv)
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
- $s_i$: supply capacity of warehouse $i$ (from supply_capacity.csv)
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
- $c_{ij}$: cost per unit from warehouse $i$ to customer $j$ (from transportation_costs.csv):

|        | C1           | C2           | C3           | C4           | C5           | C6           | C7           | C8           | C9           | C10          |
|--------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| S1     | 2077.0586725 | 0.0          | 54.33526480  | 0.0          | 0.0          | 36.17284629  | 0.0          | 0.0          | 169.33026927 | 0.0          |
| S2     | 2077.0586725 | 0.0          | 1141.0405609 | 0.0          | 0.0          | 651.11123325 | 0.0          | 0.0          | 8.06334616   | 0.0          |
| S3     | 79.92102960  | 474.24509131 | 1477.0676289 | 22.58309959  | 474.24509131 | 41.10659696  | 474.24509131 | 474.24509131 | 624.16253950 | 474.24509131 |
| S4     | 1659.3369291 | 57.20541469  | 186.15190481 | 1201.3137084 | 1029.6974644 | 41.82210594  | 57.20541469  | 1201.3137084 | 884.56338707 | 1029.6974644 |
| S5     | 1297.2567041 | 77.76629131  | 24.26760228  | 1399.7932436 | 77.76629131  | 53.91161728  | 1399.7932436 | 77.76629131  | 1255.1151480 | 1399.7932436 |
| S6     | 1998.9090659 | 985.31654357 | 2.85416869   | 1149.5359675 | 985.31654357 | 730.69236477 | 54.73980798  | 985.31654357 | 46.80310221  | 1149.5359675 |
| S7     | 1780.3360050 | 0.0          | 1141.0405609 | 0.0          | 0.0          | 36.17284629  | 0.0          | 0.0          | 8.06334616   | 0.0          |
| S8     | 75.40935896  | 1338.1987291 | 21.39134599  | 74.34437384  | 74.34437384  | 937.35062391 | 1338.1987291 | 1338.1987291 | 1392.1186581 | 1338.1987291 |
| S9     | 98.90755583  | 0.0          | 978.03476648 | 0.0          | 0.0          | 651.11123325 | 0.0          | 0.0          | 169.33026927 | 0.0          |
| S10    | 2077.0586725 | 0.0          | 54.33526480  | 0.0          | 0.0          | 36.17284629  | 0.0          | 0.0          | 145.14023080 | 0.0          |

#### Decision Variables

- $x_{ij} \geq 0$, continuous, for all warehouses $i$ and customers $j$.

#### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in \{\text{S1},\ldots,\text{S10}\}} \sum_{j \in \{\text{C1},\ldots,\text{C10}\}} c_{ij} x_{ij}
$$

#### Constraints

1. **Demand Satisfaction (for each customer $j$):**
   $$
   \sum_{i \in \{\text{S1},\ldots,\text{S10}\}} x_{ij} = d_j \qquad \forall j \in \{\text{C1},\ldots,\text{C10}\}
   $$
   Explicitly:
   - $\sum_{i} x_{i,\text{C1}} = 45$
   - $\sum_{i} x_{i,\text{C2}} = 23$
   - $\sum_{i} x_{i,\text{C3}} = 94$
   - $\sum_{i} x_{i,\text{C4}} = 92$
   - $\sum_{i} x_{i,\text{C5}} = 57$
   - $\sum_{i} x_{i,\text{C6}} = 52$
   - $\sum_{i} x_{i,\text{C7}} = 23$
   - $\sum_{i} x_{i,\text{C8}} = 99$
   - $\sum_{i} x_{i,\text{C9}} = 99$
   - $\sum_{i} x_{i,\text{C10}} = 77$

2. **Supply Capacity (for each warehouse $i$):**
   $$
   \sum_{j \in \{\text{C1},\ldots,\text{C10}\}} x_{ij} \leq s_i \qquad \forall i \in \{\text{S1},\ldots,\text{S10}\}
   $$
   Explicitly:
   - $\sum_{j} x_{\text{S1},j} \leq 127$
   - $\sum_{j} x_{\text{S2},j} \leq 236$
   - $\sum_{j} x_{\text{S3},j} \leq 168$
   - $\sum_{j} x_{\text{S4},j} \leq 115$
   - $\sum_{j} x_{\text{S5},j} \leq 280$
   - $\sum_{j} x_{\text{S6},j} \leq 179$
   - $\sum_{j} x_{\text{S7},j} \leq 135$
   - $\sum_{j} x_{\text{S8},j} \leq 263$
   - $\sum_{j} x_{\text{S9},j} \leq 283$
   - $\sum_{j} x_{\text{S10},j} \leq 476$

3. **Nonnegativity:**
   $$
   x_{ij} \geq 0 \qquad \forall i, j
   $$

#### Complete Model

Minimize
$$
\sum_{i \in \{\text{S1},\ldots,\text{S10}\}} \sum_{j \in \{\text{C1},\ldots,\text{C10}\}} c_{ij} x_{ij}
$$

Subject to
$$
\sum_{i} x_{ij} = d_j \qquad \forall j \in \{\text{C1},\ldots,\text{C10}\}
$$
$$
\sum_{j} x_{ij} \leq s_i \qquad \forall i \in \{\text{S1},\ldots,\text{S10}\}
$$
$$
x_{ij} \geq 0 \qquad \forall i, j
$$

where all $c_{ij}$, $d_j$, and $s_i$ are as given above.