##### Decision Variables:

Let $x_{ij}$ denote the quantity of beverages shipped from supply plant $i$ to customer $j$, where $i \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ and $j \in \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$.

##### Objective Function:

$\quad \min \sum_{i \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}} \sum_{j \in \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}} c_{ij} x_{ij}$

where $c_{ij}$ is the transportation cost per unit from plant $i$ to customer $j$ (see parameter table below).

##### Constraints:

###### 1. Demand Satisfaction (for each customer):

$\sum_{i \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}} x_{ij} = d_j \quad \forall j \in \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$

where $d_j$ is the demand for customer $j$.

###### 2. Supply Capacity (for each plant):

$\sum_{j \in \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}} x_{ij} \leq s_i \quad \forall i \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$

where $s_i$ is the supply capacity of plant $i$.

###### 3. Non-negativity:

$x_{ij} \geq 0 \quad \forall i, j$

---

##### Retrieved Information

```json
{
  "customers": {
    "C1": 94,
    "C2": 39,
    "C3": 65,
    "C4": 435
  },
  "plants": {
    "S1": 2531,
    "S2": 20,
    "S3": 210,
    "S4": 241
  },
  "transportation_costs": {
    "S1": {
      "C1": 543.756480860856,
      "C2": 23.685276141764653,
      "C3": 23.676386730773032,
      "C4": 447.75143678673766
    },
    "S2": {
      "C1": 883.9151090405642,
      "C2": 0.04977684765576961,
      "C3": 0.0350986687216299,
      "C4": 44.45588531711622
    },
    "S3": {
      "C1": 537.3456896658107,
      "C2": 23.769274659075112,
      "C3": 498.95659249465467,
      "C4": 440.60737890439776
    },
    "S4": {
      "C1": 1791.493192397229,
      "C2": 68.21633865655126,
      "C3": 1432.4837339656747,
      "C4": 1527.7635425462734
    }
  }
}
```

- Customers and their demands:
  - C1: 94
  - C2: 39
  - C3: 65
  - C4: 435

- Plants and their supply capacities:
  - S1: 2531
  - S2: 20
  - S3: 210
  - S4: 241

- Transportation costs per unit ($c_{ij}$):

|        |   C1    |    C2    |    C3    |    C4    |
|--------|---------|----------|----------|----------|
| **S1** | 543.756 | 23.685   | 23.676   | 447.751  |
| **S2** | 883.915 | 0.0498   | 0.0351   | 44.4559  |
| **S3** | 537.346 | 23.769   | 498.957  | 440.607  |
| **S4** | 1791.49 | 68.2163  | 1432.48  | 1527.76  |

##### Variable Domains

$x_{ij} \geq 0$ for all $i \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ and $j \in \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$

---

This model determines the optimal shipment plan from each plant to each customer, minimizing total transportation cost, while meeting all customer demands and not exceeding any plant's supply capacity.