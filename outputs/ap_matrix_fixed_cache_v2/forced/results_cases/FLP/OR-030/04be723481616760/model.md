##### Decision Variables

$x_i \geq 0$: quantity of car model $i \in I$ to fulfill (continuous).

##### Objective Function

$\max \sum_{i \in I} r_i x_i$

##### Constraints

1. Inventory constraint: $x_i \leq s_i,\quad \forall i \in I$
2. Demand constraint: $x_i \leq d_i,\quad \forall i \in I$
3. Nonnegativity: $x_i \geq 0,\quad \forall i \in I$

Where:
- $I$ is the set of all car models with classification ‘FDK57’.
- $r_i$ is the revenue per unit for model $i$.
- $s_i$ is the initial inventory for model $i$.
- $d_i$ is the demand for model $i$.

##### Parameters (retrieved from CSV)

| Model   | Revenue  | Initial Inventory | Demand |
|---------|----------|-------------------|--------|
| FDK57   | 119.144  | 30                | 200    |
| FDK57   | 119.144  | 40                | 100    |
| FDK57   | 120.144  | 50                | 150    |
| FDK57   | 121.244  | 30                | 150    |
| FDK57   | 120.544  | 10                | 150    |

So, the set $I = \{\text{FDK57, FDK57, FDK57, FDK57, FDK57}\}$, with the following parameters for each instance:

- $r_i$: [119.144, 119.144, 120.144, 121.244, 120.544]
- $s_i$: [30, 40, 50, 30, 10]
- $d_i$: [200, 100, 150, 150, 150]

##### Complete Model

Let $I = \{1,2,3,4,5\}$ index the five FDK57 entries above, with parameters as listed.

$\max\ 119.144\,x_1 + 119.144\,x_2 + 120.144\,x_3 + 121.244\,x_4 + 120.544\,x_5$

Subject to:
- $x_1 \leq 30$
- $x_2 \leq 40$
- $x_3 \leq 50$
- $x_4 \leq 30$
- $x_5 \leq 10$
- $x_1 \leq 200$
- $x_2 \leq 100$
- $x_3 \leq 150$
- $x_4 \leq 150$
- $x_5 \leq 150$
- $x_i \geq 0$ for $i=1,\ldots,5$

##### Retrieved Information

{
  "models": [
    {"Model": "FDK57", "Revenue": 119.144, "Initial Inventory": 30, "Demand": 200},
    {"Model": "FDK57", "Revenue": 119.144, "Initial Inventory": 40, "Demand": 100},
    {"Model": "FDK57", "Revenue": 120.144, "Initial Inventory": 50, "Demand": 150},
    {"Model": "FDK57", "Revenue": 121.244, "Initial Inventory": 30, "Demand": 150},
    {"Model": "FDK57", "Revenue": 120.544, "Initial Inventory": 10, "Demand": 150}
  ]
}