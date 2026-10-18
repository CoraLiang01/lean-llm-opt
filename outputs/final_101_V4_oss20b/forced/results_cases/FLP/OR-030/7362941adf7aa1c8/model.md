##### Decision Variables

$x_i \ge 0$: quantity of car model $i \in I$ sold.

##### Objective Function

$\max \sum_{i \in I} r_i x_i$

##### Constraints

1. Inventory: $\sum_{i \in I}x_i \le I_i,\quad \forall i \in I$.
2. Demand: $\sum_{i \in I}x_i \le D_i,\quad \forall i \in I$.
3. Domains: $x_i \ge 0$.

$I = \{ \text{FDK57-001}, \text{FDK57-002}, \text{FDK57-003}, \text{FDK57-004}, \text{FDK57-005}, \text{FDK57-006}, \text{FDK57-007}, \text{FDK57-008} \}$

###### Retrieved Information

{
  "car_models": [
    "FDK57-001",
    "FDK57-002",
    "FDK57-003",
    "FDK57-004",
    "FDK57-005",
    "FDK57-006",
    "FDK57-007",
    "FDK57-008"
  ],
  "revenue": [
    119.144,
    119.144,
    121.244,
    120.144,
    120.544,
    121.244,
    119.744,
    120.844
  ],
  "initial_inventory": [
    200,
    100,
    200,
    150,
    150,
    150,
    250,
    150
  ],
  "demand": [
    30,
    40,
    30,
    50,
    10,
    30,
    50,
    50
  ]
}