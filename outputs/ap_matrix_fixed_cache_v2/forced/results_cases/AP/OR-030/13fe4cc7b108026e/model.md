##### Decision Variables:

Let $x_i$ denote the quantity of ‘FDK57’ car model $i$ to fulfill, for $i = 1,2,3,4,5$ (corresponding to the five rows below).

##### Parameters:

- Revenue per unit: $r_i$
- Initial Inventory: $I_i$
- Demand: $d_i$

| $i$ | Product Name | $r_i$ (Revenue) | $I_i$ (Initial Inventory) | $d_i$ (Demand) |
|-----|--------------|-----------------|--------------------------|----------------|
| 1   | FDK57        | 119.144         | 200                      | 30             |
| 2   | FDK57        | 119.144         | 100                      | 40             |
| 3   | FDK57        | 120.144         | 150                      | 50             |
| 4   | FDK57        | 121.244         | 200                      | 30             |
| 5   | FDK57        | 120.544         | 150                      | 10             |

##### Objective Function:

$\quad \max \sum_{i=1}^5 r_i x_i$

That is,

$\quad \max \left(119.144\,x_1 + 119.144\,x_2 + 120.144\,x_3 + 121.244\,x_4 + 120.544\,x_5\right)$

##### Constraints:

1. Inventory and Demand Fulfillment:

$\quad 0 \leq x_i \leq \min\{I_i, d_i\} \quad \forall i = 1,2,3,4,5$

That is,

$\quad 0 \leq x_1 \leq 30$

$\quad 0 \leq x_2 \leq 40$

$\quad 0 \leq x_3 \leq 50$

$\quad 0 \leq x_4 \leq 30$

$\quad 0 \leq x_5 \leq 10$

2. (Optional, if integer fulfillment is required):

$\quad x_i \in \mathbb{Z}_{\geq 0} \quad \forall i$

##### Retrieved Information

{
  "models": [
    {
      "Product Name": "FDK57",
      "Revenue": 119.144,
      "Demand": 30,
      "Initial Inventory": 200
    },
    {
      "Product Name": "FDK57",
      "Revenue": 119.144,
      "Demand": 40,
      "Initial Inventory": 100
    },
    {
      "Product Name": "FDK57",
      "Revenue": 120.144,
      "Demand": 50,
      "Initial Inventory": 150
    },
    {
      "Product Name": "FDK57",
      "Revenue": 121.244,
      "Demand": 30,
      "Initial Inventory": 200
    },
    {
      "Product Name": "FDK57",
      "Revenue": 120.544,
      "Demand": 10,
      "Initial Inventory": 150
    }
  ]
}