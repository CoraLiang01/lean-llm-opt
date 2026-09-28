##### Decision Variables

Let $x_i$ denote the number of units of product $i$ (where $i$ indexes the set of "FAUX" products) to be fulfilled.

##### Objective Function

$\max \sum_{i=1}^{12} r_i x_i$

where $r_i$ is the revenue per unit for product $i$.

##### Constraints

1. **Inventory and Demand Constraints:**

For each product $i$:

$0 \leq x_i \leq \min\{\text{Initial Inventory}_i, \text{Demand}_i\}$

or, equivalently, two constraints per product:

$x_i \leq \text{Initial Inventory}_i$

$x_i \leq \text{Demand}_i$

$x_i \geq 0$

2. **Variable Domain:**

$x_i$ are integer variables (if only whole units can be fulfilled), or continuous and non-negative if partial units are allowed.

##### Retrieved Information

{
  "products": [
    {
      "name": "FAUX FUR JEWEL SWEATER",
      "revenue": 35.9,
      "initial_inventory": 20970,
      "demand": 3025
    },
    {
      "name": "FAUX LEATHER BOMBER JACKET",
      "revenue": 69.9,
      "initial_inventory": 71970,
      "demand": 9585
    },
    {
      "name": "FAUX LEATHER BOXY FIT JACKET",
      "revenue": 99.9,
      "initial_inventory": 32730,
      "demand": 4486
    },
    {
      "name": "FAUX LEATHER JACKET",
      "revenue": 99.9,
      "initial_inventory": 71130,
      "demand": 10322
    },
    {
      "name": "FAUX LEATHER OVERSIZED JACKET LIMITED EDITION",
      "revenue": 159.0,
      "initial_inventory": 34910,
      "demand": 4868
    },
    {
      "name": "FAUX LEATHER PUFFER JACKET",
      "revenue": 69.99,
      "initial_inventory": 64010,
      "demand": 8482
    },
    {
      "name": "FAUX SHEARLING LINED SUEDE BOOTS",
      "revenue": 99.9,
      "initial_inventory": 20760,
      "demand": 2607
    },
    {
      "name": "FAUX SHEARLING PLAID JACKET",
      "revenue": 89.9,
      "initial_inventory": 12490,
      "demand": 1784
    },
    {
      "name": "FAUX SUEDE BOMBER JACKET",
      "revenue": 69.9,
      "initial_inventory": 50300,
      "demand": 6626
    },
    {
      "name": "FAUX SUEDE JACKET",
      "revenue": 89.9,
      "initial_inventory": 24570,
      "demand": 3256
    },
    {
      "name": "FAUX SUEDE OVERSHIRT",
      "revenue": 69.9,
      "initial_inventory": 24430,
      "demand": 2955
    },
    {
      "name": "FAUX SUEDE PATCH JACKET",
      "revenue": 89.9,
      "initial_inventory": 7070,
      "demand": 910
    }
  ]
}

##### Full Model (with explicit parameters):

Let the set of products $i = 1, \ldots, 12$ correspond to the following:

| $i$ | Product Name                                      | $r_i$   | Initial Inventory | Demand |
|-----|---------------------------------------------------|---------|------------------|--------|
| 1   | FAUX FUR JEWEL SWEATER                            | 35.9    | 20970            | 3025   |
| 2   | FAUX LEATHER BOMBER JACKET                        | 69.9    | 71970            | 9585   |
| 3   | FAUX LEATHER BOXY FIT JACKET                      | 99.9    | 32730            | 4486   |
| 4   | FAUX LEATHER JACKET                               | 99.9    | 71130            | 10322  |
| 5   | FAUX LEATHER OVERSIZED JACKET LIMITED EDITION     | 159.0   | 34910            | 4868   |
| 6   | FAUX LEATHER PUFFER JACKET                        | 69.99   | 64010            | 8482   |
| 7   | FAUX SHEARLING LINED SUEDE BOOTS                  | 99.9    | 20760            | 2607   |
| 8   | FAUX SHEARLING PLAID JACKET                       | 89.9    | 12490            | 1784   |
| 9   | FAUX SUEDE BOMBER JACKET                          | 69.9    | 50300            | 6626   |
| 10  | FAUX SUEDE JACKET                                 | 89.9    | 24570            | 3256   |
| 11  | FAUX SUEDE OVERSHIRT                              | 69.9    | 24430            | 2955   |
| 12  | FAUX SUEDE PATCH JACKET                           | 89.9    | 7070             | 910    |

$\max \left(35.9 x_1 + 69.9 x_2 + 99.9 x_3 + 99.9 x_4 + 159.0 x_5 + 69.99 x_6 + 99.9 x_7 + 89.9 x_8 + 69.9 x_9 + 89.9 x_{10} + 69.9 x_{11} + 89.9 x_{12}\right)$

Subject to, for $i = 1, \ldots, 12$:

$x_i \leq$ Initial Inventory$_i$

$x_i \leq$ Demand$_i$

$x_i \geq 0$

and $x_i$ integer (if only whole units can be fulfilled).