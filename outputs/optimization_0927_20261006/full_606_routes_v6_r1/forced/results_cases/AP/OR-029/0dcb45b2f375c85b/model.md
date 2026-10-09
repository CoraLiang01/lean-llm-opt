##### Decision Variables

Let $x_i$ denote the number of units of product $i$ (where $i$ indexes the set of ‘FAUX’ products) to be fulfilled.

##### Objective Function

$\max \sum_{i} r_i x_i$

where $r_i$ is the revenue per unit for product $i$.

##### Constraints

For each product $i$:

$0 \leq x_i \leq \min\{\text{Initial Inventory}_i, \text{Demand}_i\}$

Or, equivalently, separated:

$x_i \leq \text{Initial Inventory}_i$

$x_i \leq \text{Demand}_i$

$x_i \geq 0$

##### Retrieved Information

{
  "products": [
    {
      "Product Name": "FAUX FUR JEWEL SWEATER",
      "Revenue": 35.9,
      "Initial Inventory": 20970,
      "Demand": 3025
    },
    {
      "Product Name": "FAUX LEATHER BOMBER JACKET",
      "Revenue": 69.9,
      "Initial Inventory": 71970,
      "Demand": 9585
    },
    {
      "Product Name": "FAUX LEATHER BOXY FIT JACKET",
      "Revenue": 99.9,
      "Initial Inventory": 32730,
      "Demand": 4486
    },
    {
      "Product Name": "FAUX LEATHER JACKET",
      "Revenue": 99.9,
      "Initial Inventory": 71130,
      "Demand": 10322
    },
    {
      "Product Name": "FAUX LEATHER OVERSIZED JACKET LIMITED EDITION",
      "Revenue": 159.0,
      "Initial Inventory": 34910,
      "Demand": 4868
    },
    {
      "Product Name": "FAUX LEATHER PUFFER JACKET",
      "Revenue": 69.99,
      "Initial Inventory": 64010,
      "Demand": 8482
    },
    {
      "Product Name": "FAUX SHEARLING LINED SUEDE BOOTS",
      "Revenue": 99.9,
      "Initial Inventory": 20760,
      "Demand": 2607
    },
    {
      "Product Name": "FAUX SHEARLING PLAID JACKET",
      "Revenue": 89.9,
      "Initial Inventory": 12490,
      "Demand": 1784
    },
    {
      "Product Name": "FAUX SUEDE BOMBER JACKET",
      "Revenue": 69.9,
      "Initial Inventory": 50300,
      "Demand": 6626
    },
    {
      "Product Name": "FAUX SUEDE JACKET",
      "Revenue": 89.9,
      "Initial Inventory": 24570,
      "Demand": 3256
    },
    {
      "Product Name": "FAUX SUEDE OVERSHIRT",
      "Revenue": 69.9,
      "Initial Inventory": 24430,
      "Demand": 2955
    },
    {
      "Product Name": "FAUX SUEDE PATCH JACKET",
      "Revenue": 89.9,
      "Initial Inventory": 7070,
      "Demand": 910
    }
  ]
}

##### Explicit Model

Let the set of products $i = 1, \ldots, 12$ correspond to the following:

1. FAUX FUR JEWEL SWEATER
2. FAUX LEATHER BOMBER JACKET
3. FAUX LEATHER BOXY FIT JACKET
4. FAUX LEATHER JACKET
5. FAUX LEATHER OVERSIZED JACKET LIMITED EDITION
6. FAUX LEATHER PUFFER JACKET
7. FAUX SHEARLING LINED SUEDE BOOTS
8. FAUX SHEARLING PLAID JACKET
9. FAUX SUEDE BOMBER JACKET
10. FAUX SUEDE JACKET
11. FAUX SUEDE OVERSHIRT
12. FAUX SUEDE PATCH JACKET

With parameters:

| $i$ | Product Name                                   | $r_i$   | Initial Inventory | Demand |
|-----|-----------------------------------------------|---------|------------------|--------|
| 1   | FAUX FUR JEWEL SWEATER                        | 35.9    | 20970            | 3025   |
| 2   | FAUX LEATHER BOMBER JACKET                    | 69.9    | 71970            | 9585   |
| 3   | FAUX LEATHER BOXY FIT JACKET                  | 99.9    | 32730            | 4486   |
| 4   | FAUX LEATHER JACKET                           | 99.9    | 71130            | 10322  |
| 5   | FAUX LEATHER OVERSIZED JACKET LIMITED EDITION | 159.0   | 34910            | 4868   |
| 6   | FAUX LEATHER PUFFER JACKET                    | 69.99   | 64010            | 8482   |
| 7   | FAUX SHEARLING LINED SUEDE BOOTS              | 99.9    | 20760            | 2607   |
| 8   | FAUX SHEARLING PLAID JACKET                   | 89.9    | 12490            | 1784   |
| 9   | FAUX SUEDE BOMBER JACKET                      | 69.9    | 50300            | 6626   |
| 10  | FAUX SUEDE JACKET                             | 89.9    | 24570            | 3256   |
| 11  | FAUX SUEDE OVERSHIRT                          | 69.9    | 24430            | 2955   |
| 12  | FAUX SUEDE PATCH JACKET                       | 89.9    | 7070             | 910    |

The model:

$\max \left(35.9\,x_1 + 69.9\,x_2 + 99.9\,x_3 + 99.9\,x_4 + 159.0\,x_5 + 69.99\,x_6 + 99.9\,x_7 + 89.9\,x_8 + 69.9\,x_9 + 89.9\,x_{10} + 69.9\,x_{11} + 89.9\,x_{12}\right)$

Subject to, for each $i$:

$0 \leq x_1 \leq \min\{20970, 3025\} = 3025$

$0 \leq x_2 \leq \min\{71970, 9585\} = 9585$

$0 \leq x_3 \leq \min\{32730, 4486\} = 4486$

$0 \leq x_4 \leq \min\{71130, 10322\} = 10322$

$0 \leq x_5 \leq \min\{34910, 4868\} = 4868$

$0 \leq x_6 \leq \min\{64010, 8482\} = 8482$

$0 \leq x_7 \leq \min\{20760, 2607\} = 2607$

$0 \leq x_8 \leq \min\{12490, 1784\} = 1784$

$0 \leq x_9 \leq \min\{50300, 6626\} = 6626$

$0 \leq x_{10} \leq \min\{24570, 3256\} = 3256$

$0 \leq x_{11} \leq \min\{24430, 2955\} = 2955$

$0 \leq x_{12} \leq \min\{7070, 910\} = 910$

And all $x_i$ are continuous and non-negative.