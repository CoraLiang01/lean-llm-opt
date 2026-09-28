##### Decision Variables

Let $x_i$ denote the number of units of product $i$ (where $i$ indexes the set of ‘ELE-S’ products) to fulfill.

##### Objective Function

$\quad \max \sum_{i=1}^8 r_i x_i$

where $r_i$ is the revenue per unit for product $i$.

##### Constraints

For each product $i$:

$0 \leq x_i \leq \min\{\text{Initial Inventory}_i, \text{Demand}_i\}$

or, equivalently, for all $i$:

$x_i \leq \text{Initial Inventory}_i$

$x_i \leq \text{Demand}_i$

$x_i \geq 0$

$x_i \in \mathbb{Z}$

##### Retrieved Information

{
  "products": [
    {
      "Product Reference": "ELE-SMA-10000463",
      "Revenue": 4.0,
      "Initial Inventory": 2000.0,
      "Demand": 295
    },
    {
      "Product Reference": "ELE-SMA-10000487",
      "Revenue": 14.0,
      "Initial Inventory": 7000.0,
      "Demand": 1002
    },
    {
      "Product Reference": "ELE-SMA-10003333",
      "Revenue": 14.0,
      "Initial Inventory": 7000.0,
      "Demand": 958
    },
    {
      "Product Reference": "ELE-SMA-10009012",
      "Revenue": 4.0,
      "Initial Inventory": 6000.0,
      "Demand": 777
    },
    {
      "Product Reference": "ELE-SMA-10009999",
      "Revenue": 4.0,
      "Initial Inventory": 2000.0,
      "Demand": 271
    },
    {
      "Product Reference": "ELE-SMA-10011234",
      "Revenue": 4.0,
      "Initial Inventory": 2000.0,
      "Demand": 244
    },
    {
      "Product Reference": "ELE-SMA-10027456",
      "Revenue": 14.0,
      "Initial Inventory": 7000.0,
      "Demand": 990
    },
    {
      "Product Reference": "ELE-SMA-10028567",
      "Revenue": 14.0,
      "Initial Inventory": 7000.0,
      "Demand": 1000
    }
  ]
}

##### Model Parameters

Let the index set $i = 1, \ldots, 8$ correspond to the products in the order listed above.

- $r = [4.0, 14.0, 14.0, 4.0, 4.0, 4.0, 14.0, 14.0]$
- $\text{Initial Inventory} = [2000, 7000, 7000, 6000, 2000, 2000, 7000, 7000]$
- $\text{Demand} = [295, 1002, 958, 777, 271, 244, 990, 1000]$

##### Complete Mathematical Model

$\max \left(4.0\,x_1 + 14.0\,x_2 + 14.0\,x_3 + 4.0\,x_4 + 4.0\,x_5 + 4.0\,x_6 + 14.0\,x_7 + 14.0\,x_8\right)$

subject to:

$x_1 \leq 2000$

$x_1 \leq 295$

$x_2 \leq 7000$

$x_2 \leq 1002$

$x_3 \leq 7000$

$x_3 \leq 958$

$x_4 \leq 6000$

$x_4 \leq 777$

$x_5 \leq 2000$

$x_5 \leq 271$

$x_6 \leq 2000$

$x_6 \leq 244$

$x_7 \leq 7000$

$x_7 \leq 990$

$x_8 \leq 7000$

$x_8 \leq 1000$

$x_i \geq 0,\quad x_i \in \mathbb{Z},\quad \forall i=1,\ldots,8$

##### Product Index Mapping

1. $x_1$: ELE-SMA-10000463
2. $x_2$: ELE-SMA-10000487
3. $x_3$: ELE-SMA-10003333
4. $x_4$: ELE-SMA-10009012
5. $x_5$: ELE-SMA-10009999
6. $x_6$: ELE-SMA-10011234
7. $x_7$: ELE-SMA-10027456
8. $x_8$: ELE-SMA-10028567