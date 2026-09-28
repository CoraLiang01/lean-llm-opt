##### Decision Variables:

Let $x_i$ denote the quantity of car model $i$ (for each row below) that the dealership plans to fulfill.

##### Parameters:

- $r_i$: Revenue per unit for car model $i$
- $I_i$: Initial inventory for car model $i$
- $d_i$: Demand for car model $i$

##### Objective Function:

$\quad \max \sum_{i=1}^5 r_i x_i$

##### Constraints:

1. Inventory and Demand Constraints:

$\quad 0 \leq x_i \leq \min\{I_i, d_i\} \quad \forall i = 1,2,3,4,5$

##### Variable Constraints:

$\quad x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1,2,3,4,5$

##### Retrieved Information

{
  "models": [
    {
      "Model": "FDK57",
      "Revenue": 119.144,
      "Initial Inventory": 200,
      "Demand": 30
    },
    {
      "Model": "FDK57",
      "Revenue": 119.144,
      "Initial Inventory": 100,
      "Demand": 40
    },
    {
      "Model": "FDK57",
      "Revenue": 120.144,
      "Initial Inventory": 150,
      "Demand": 50
    },
    {
      "Model": "FDK57",
      "Revenue": 121.244,
      "Initial Inventory": 200,
      "Demand": 30
    },
    {
      "Model": "FDK57",
      "Revenue": 120.544,
      "Initial Inventory": 150,
      "Demand": 10
    }
  ]
}

##### Explicit Model

Let $x_1, x_2, x_3, x_4, x_5$ correspond to the five rows above, in order.

Objective:
$$
\max \left(119.144\, x_1 + 119.144\, x_2 + 120.144\, x_3 + 121.244\, x_4 + 120.544\, x_5\right)
$$

Subject to:
$$
0 \leq x_1 \leq 30 \\
0 \leq x_2 \leq 40 \\
0 \leq x_3 \leq 50 \\
0 \leq x_4 \leq 30 \\
0 \leq x_5 \leq 10 \\
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1,2,3,4,5
$$