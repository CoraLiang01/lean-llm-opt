##### Sets and Indices

Let $I$ be the set of all products classified as ‘FAUX’:
\[
I = \{
\text{FAUX FUR JEWEL SWEATER},
\text{FAUX LEATHER BOMBER JACKET},
\text{FAUX LEATHER BOXY FIT JACKET},
\text{FAUX LEATHER JACKET},
\text{FAUX LEATHER OVERSIZED JACKET LIMITED EDITION},
\text{FAUX LEATHER PUFFER JACKET},
\text{FAUX SHEARLING LINED SUEDE BOOTS},
\text{FAUX SHEARLING PLAID JACKET},
\text{FAUX SUEDE BOMBER JACKET},
\text{FAUX SUEDE JACKET},
\text{FAUX SUEDE OVERSHIRT},
\text{FAUX SUEDE PATCH JACKET}
\}
\]

##### Parameters

For each $i \in I$:

- $r_i$ = Revenue per unit of product $i$
- $d_i$ = Demand for product $i$
- $s_i$ = Initial Inventory of product $i$

\[
\begin{array}{|l|c|c|c|}
\hline
\text{Product Name} & r_i & d_i & s_i \\
\hline
\text{FAUX FUR JEWEL SWEATER} & 35.9 & 3025 & 20970 \\
\text{FAUX LEATHER BOMBER JACKET} & 69.9 & 9585 & 71970 \\
\text{FAUX LEATHER BOXY FIT JACKET} & 99.9 & 4486 & 32730 \\
\text{FAUX LEATHER JACKET} & 99.9 & 10322 & 71130 \\
\text{FAUX LEATHER OVERSIZED JACKET LIMITED EDITION} & 159.0 & 4868 & 34910 \\
\text{FAUX LEATHER PUFFER JACKET} & 69.99 & 8482 & 64010 \\
\text{FAUX SHEARLING LINED SUEDE BOOTS} & 99.9 & 2607 & 20760 \\
\text{FAUX SHEARLING PLAID JACKET} & 89.9 & 1784 & 12490 \\
\text{FAUX SUEDE BOMBER JACKET} & 69.9 & 6626 & 50300 \\
\text{FAUX SUEDE JACKET} & 89.9 & 3256 & 24570 \\
\text{FAUX SUEDE OVERSHIRT} & 69.9 & 2955 & 24430 \\
\text{FAUX SUEDE PATCH JACKET} & 89.9 & 910 & 7070 \\
\hline
\end{array}
\]

##### Decision Variables

For each $i \in I$:

- $x_i$ = Number of units of product $i$ to fulfill (integer, $x_i \geq 0$)

##### Objective Function

\[
\max \sum_{i \in I} r_i x_i
\]

##### Constraints

For each $i \in I$:

1. Inventory constraint:
   \[
   x_i \leq s_i
   \]
2. Demand constraint:
   \[
   x_i \leq d_i
   \]
3. Non-negativity and integrality:
   \[
   x_i \geq 0, \quad x_i \in \mathbb{Z}
   \]

##### Retrieved Information

{
  "products": [
    {
      "Product Name": "FAUX FUR JEWEL SWEATER",
      "Revenue": 35.9,
      "Demand": 3025,
      "Initial Inventory": 20970
    },
    {
      "Product Name": "FAUX LEATHER BOMBER JACKET",
      "Revenue": 69.9,
      "Demand": 9585,
      "Initial Inventory": 71970
    },
    {
      "Product Name": "FAUX LEATHER BOXY FIT JACKET",
      "Revenue": 99.9,
      "Demand": 4486,
      "Initial Inventory": 32730
    },
    {
      "Product Name": "FAUX LEATHER JACKET",
      "Revenue": 99.9,
      "Demand": 10322,
      "Initial Inventory": 71130
    },
    {
      "Product Name": "FAUX LEATHER OVERSIZED JACKET LIMITED EDITION",
      "Revenue": 159.0,
      "Demand": 4868,
      "Initial Inventory": 34910
    },
    {
      "Product Name": "FAUX LEATHER PUFFER JACKET",
      "Revenue": 69.99,
      "Demand": 8482,
      "Initial Inventory": 64010
    },
    {
      "Product Name": "FAUX SHEARLING LINED SUEDE BOOTS",
      "Revenue": 99.9,
      "Demand": 2607,
      "Initial Inventory": 20760
    },
    {
      "Product Name": "FAUX SHEARLING PLAID JACKET",
      "Revenue": 89.9,
      "Demand": 1784,
      "Initial Inventory": 12490
    },
    {
      "Product Name": "FAUX SUEDE BOMBER JACKET",
      "Revenue": 69.9,
      "Demand": 6626,
      "Initial Inventory": 50300
    },
    {
      "Product Name": "FAUX SUEDE JACKET",
      "Revenue": 89.9,
      "Demand": 3256,
      "Initial Inventory": 24570
    },
    {
      "Product Name": "FAUX SUEDE OVERSHIRT",
      "Revenue": 69.9,
      "Demand": 2955,
      "Initial Inventory": 24430
    },
    {
      "Product Name": "FAUX SUEDE PATCH JACKET",
      "Revenue": 89.9,
      "Demand": 910,
      "Initial Inventory": 7070
    }
  ]
}