##### Objective Function: 

$ \max \sum_{i=1}^{12} r_i x_i $

##### Constraints

###### 1. Demand and Inventory Constraints: 

$ 0 \le x_i \le \min(d_i,\; I_i) \quad \forall i \in \{1,\dots,12\} $

###### 2. Integer Constraints: 

$ x_i \in \mathbb{Z}_{+} \quad \forall i \in \{1,\dots,12\} $

##### Retrieved Information

{
  "products": [
    {
      "name": "FAUX FUR JEWEL SWEATER",
      "revenue": 35.9,
      "demand": 3025,
      "initial_inventory": 20970
    },
    {
      "name": "FAUX LEATHER BOMBER JACKET",
      "revenue": 69.9,
      "demand": 9585,
      "initial_inventory": 71970
    },
    {
      "name": "FAUX LEATHER BOXY FIT JACKET",
      "revenue": 99.9,
      "demand": 4486,
      "initial_inventory": 32730
    },
    {
      "name": "FAUX LEATHER JACKET",
      "revenue": 99.9,
      "demand": 10322,
      "initial_inventory": 71130
    },
    {
      "name": "FAUX LEATHER OVERSIZED JACKET LIMITED EDITION",
      "revenue": 159.0,
      "demand": 4868,
      "initial_inventory": 34910
    },
    {
      "name": "FAUX LEATHER PUFFER JACKET",
      "revenue": 69.99,
      "demand": 8482,
      "initial_inventory": 64010
    },
    {
      "name": "FAUX SHEARLING LINED SUEDE BOOTS",
      "revenue": 99.9,
      "demand": 2607,
      "initial_inventory": 20760
    },
    {
      "name": "FAUX SHEARLING PLAID JACKET",
      "revenue": 89.9,
      "demand": 1784,
      "initial_inventory": 12490
    },
    {
      "name": "FAUX SUEDE BOMBER JACKET",
      "revenue": 69.9,
      "demand": 6626,
      "initial_inventory": 50300
    },
    {
      "name": "FAUX SUEDE JACKET",
      "revenue": 89.9,
      "demand": 3256,
      "initial_inventory": 24570
    },
    {
      "name": "FAUX SUEDE OVERSHIRT",
      "revenue": 69.9,
      "demand": 2955,
      "initial_inventory": 24430
    },
    {
      "name": "FAUX SUEDE PATCH JACKET",
      "revenue": 89.9,
      "demand": 910,
      "initial_inventory": 7070
    }
  ],
  "vectors": {
    "revenue": [35.9, 69.9, 99.9, 99.9, 159.0, 69.99, 99.9, 89.9, 69.9, 89.9, 69.9, 89.9],
    "demand": [3025, 9585, 4486, 10322, 4868, 8482, 2607, 1784, 6626, 3256, 2955, 910],
    "initial_inventory": [20970, 71970, 32730, 71130, 34910, 64010, 20760, 12490, 50300, 24570, 24430, 7070]
  }
}