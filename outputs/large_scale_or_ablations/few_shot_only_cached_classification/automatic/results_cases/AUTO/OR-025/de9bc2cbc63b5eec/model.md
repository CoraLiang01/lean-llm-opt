##### Sets and Indices

Let $i$ index all products classified as ‘TABLET’ in the source order below.

##### Parameters (from retrieved data)

For each $i$:

- $A_i$: Revenue per unit of product $i$
- $d_i$: Demand for product $i$
- $I_i$: Initial Inventory for product $i$

| Product Name         | $A_i$ (Revenue) | $d_i$ (Demand) | $I_i$ (Initial Inventory) |
|----------------------|-----------------|---------------|--------------------------|
| TABLET_10084.74      | 10084.74        | 2             | 10                       |
| TABLET_12211.86      | 12211.86        | 43            | 300                      |
| TABLET_14669.5       | 14669.5         | 6             | 30                       |
| TABLET_14745.76      | 14745.76        | 6             | 30                       |
| TABLET_14754.24      | 14754.24        | 20            | 100                      |
| TABLET_16448.3       | 16448.3         | 22            | 110                      |
| TABLET_16448.31      | 16448.31        | 3             | 20                       |
| TABLET_20143.22      | 20143.22        | 16            | 80                       |
| TABLET_2042.38       | 2042.38         | 2             | 10                       |
| TABLET_24915.25      | 24915.25        | 2             | 10                       |
| TABLET_24915.26      | 24915.26        | 32            | 160                      |
| TABLET_26448.3       | 26448.3         | 14            | 70                       |
| TABLET_27042.38      | 27042.38        | 2             | 10                       |
| TABLET_30000.0       | 30000.0         | 2             | 10                       |
| TABLET_33397.46      | 33397.46        | 6             | 30                       |
| TABLET_33398.3       | 33398.3         | 2             | 10                       |
| TABLET_48567.8       | 48567.8         | 6             | 30                       |
| TABLET_48644.07      | 48644.07        | 2             | 10                       |
| TABLET_50262.72      | 50262.72        | 6             | 30                       |
| TABLET_53736.44      | 53736.44        | 6             | 30                       |
| TABLET_6957.62       | 6957.62         | 6             | 40                       |
| TABLET_6957.63       | 6957.63         | 12            | 80                       |
| TABLET_7550.84       | 7550.84         | 60            | 300                      |
| TABLET_7550.85       | 7550.85         | 8             | 40                       |
| TABLET_9584.74       | 9584.74         | 8             | 40                       |
| TABLET_9661.02       | 9661.02         | 38            | 190                      |
| TABLET_9669.5        | 9669.5          | 4             | 20                       |

##### Decision Variables

For each ‘TABLET’ product $i$:

- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers)

##### Mathematical Model

Objective:
$$
\max \sum_{i} A_i \cdot x_i
$$

Subject to:

Inventory and demand constraints:
$$
0 \leq x_i \leq \min\{d_i, I_i\} \qquad \forall i
$$

Variable domain:
$$
x_i \in \mathbb{Z}_+ \qquad \forall i
$$

##### Retrieved Information

{
  "products": [
    {"Product Name": "TABLET_10084.74", "Revenue": 10084.74, "Demand": 2, "Initial Inventory": 10},
    {"Product Name": "TABLET_12211.86", "Revenue": 12211.86, "Demand": 43, "Initial Inventory": 300},
    {"Product Name": "TABLET_14669.5", "Revenue": 14669.5, "Demand": 6, "Initial Inventory": 30},
    {"Product Name": "TABLET_14745.76", "Revenue": 14745.76, "Demand": 6, "Initial Inventory": 30},
    {"Product Name": "TABLET_14754.24", "Revenue": 14754.24, "Demand": 20, "Initial Inventory": 100},
    {"Product Name": "TABLET_16448.3", "Revenue": 16448.3, "Demand": 22, "Initial Inventory": 110},
    {"Product Name": "TABLET_16448.31", "Revenue": 16448.31, "Demand": 3, "Initial Inventory": 20},
    {"Product Name": "TABLET_20143.22", "Revenue": 20143.22, "Demand": 16, "Initial Inventory": 80},
    {"Product Name": "TABLET_2042.38", "Revenue": 2042.38, "Demand": 2, "Initial Inventory": 10},
    {"Product Name": "TABLET_24915.25", "Revenue": 24915.25, "Demand": 2, "Initial Inventory": 10},
    {"Product Name": "TABLET_24915.26", "Revenue": 24915.26, "Demand": 32, "Initial Inventory": 160},
    {"Product Name": "TABLET_26448.3", "Revenue": 26448.3, "Demand": 14, "Initial Inventory": 70},
    {"Product Name": "TABLET_27042.38", "Revenue": 27042.38, "Demand": 2, "Initial Inventory": 10},
    {"Product Name": "TABLET_30000.0", "Revenue": 30000.0, "Demand": 2, "Initial Inventory": 10},
    {"Product Name": "TABLET_33397.46", "Revenue": 33397.46, "Demand": 6, "Initial Inventory": 30},
    {"Product Name": "TABLET_33398.3", "Revenue": 33398.3, "Demand": 2, "Initial Inventory": 10},
    {"Product Name": "TABLET_48567.8", "Revenue": 48567.8, "Demand": 6, "Initial Inventory": 30},
    {"Product Name": "TABLET_48644.07", "Revenue": 48644.07, "Demand": 2, "Initial Inventory": 10},
    {"Product Name": "TABLET_50262.72", "Revenue": 50262.72, "Demand": 6, "Initial Inventory": 30},
    {"Product Name": "TABLET_53736.44", "Revenue": 53736.44, "Demand": 6, "Initial Inventory": 30},
    {"Product Name": "TABLET_6957.62", "Revenue": 6957.62, "Demand": 6, "Initial Inventory": 40},
    {"Product Name": "TABLET_6957.63", "Revenue": 6957.63, "Demand": 12, "Initial Inventory": 80},
    {"Product Name": "TABLET_7550.84", "Revenue": 7550.84, "Demand": 60, "Initial Inventory": 300},
    {"Product Name": "TABLET_7550.85", "Revenue": 7550.85, "Demand": 8, "Initial Inventory": 40},
    {"Product Name": "TABLET_9584.74", "Revenue": 9584.74, "Demand": 8, "Initial Inventory": 40},
    {"Product Name": "TABLET_9661.02", "Revenue": 9661.02, "Demand": 38, "Initial Inventory": 190},
    {"Product Name": "TABLET_9669.5", "Revenue": 9669.5, "Demand": 4, "Initial Inventory": 20}
  ]
}