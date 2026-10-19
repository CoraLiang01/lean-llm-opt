##### Objective Function: 

$\quad \quad \max \sum_i \text{Revenue}_i \cdot x_i$

##### Constraints

###### 1. Demand Constraints: 

$\quad \quad x_i \leq \text{Demand}_i, \quad \forall i$

###### 2. Inventory Constraints: 

$\quad \quad x_i \leq \text{Initial Inventory}_i, \quad \forall i$

###### 3. Non-negativity and Integrality Constraints: 

$\quad \quad x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i$

I indexes the FAUX products; $x_i$ is the number of units of product $i$ that will be fulfilled. $\text{Revenue}_i$ is the revenue per unit, $\text{Demand}_i$ is the deterministic demand, and $\text{Initial Inventory}_i$ is the available inventory for product $i$.

###### Retrieved Information

{
  "items": [
    "FAUX FUR JEWEL SWEATER",
    "FAUX LEATHER BOMBER JACKET",
    "FAUX LEATHER BOXY FIT JACKET",
    "FAUX LEATHER JACKET",
    "FAUX LEATHER OVERSIZED JACKET LIMITED EDITION",
    "FAUX LEATHER PUFFER JACKET",
    "FAUX SHEARLING LINED SUEDE BOOTS",
    "FAUX SHEARLING PLAID JACKET",
    "FAUX SUEDE BOMBER JACKET",
    "FAUX SUEDE JACKET",
    "FAUX SUEDE OVERSHIRT",
    "FAUX SUEDE PATCH JACKET"
  ],
  "profit": {
    "FAUX FUR JEWEL SWEATER": 35.9,
    "FAUX LEATHER BOMBER JACKET": 69.9,
    "FAUX LEATHER BOXY FIT JACKET": 99.9,
    "FAUX LEATHER JACKET": 99.9,
    "FAUX LEATHER OVERSIZED JACKET LIMITED EDITION": 159.0,
    "FAUX LEATHER PUFFER JACKET": 69.99,
    "FAUX SHEARLING LINED SUEDE BOOTS": 99.9,
    "FAUX SHEARLING PLAID JACKET": 89.9,
    "FAUX SUEDE BOMBER JACKET": 69.9,
    "FAUX SUEDE JACKET": 89.9,
    "FAUX SUEDE OVERSHIRT": 69.9,
    "FAUX SUEDE PATCH JACKET": 89.9
  },
  "demand": {
    "FAUX FUR JEWEL SWEATER": 3025,
    "FAUX LEATHER BOMBER JACKET": 9585,
    "FAUX LEATHER BOXY FIT JACKET": 4486,
    "FAUX LEATHER JACKET": 10322,
    "FAUX LEATHER OVERSIZED JACKET LIMITED EDITION": 4868,
    "FAUX LEATHER PUFFER JACKET": 8482,
    "FAUX SHEARLING LINED SUEDE BOOTS": 2607,
    "FAUX SHEARLING PLAID JACKET": 1784,
    "FAUX SUEDE BOMBER JACKET": 6626,
    "FAUX SUEDE JACKET": 3256,
    "FAUX SUEDE OVERSHIRT": 2955,
    "FAUX SUEDE PATCH JACKET": 910
  },
  "inventory": {
    "FAUX FUR JEWEL SWEATER": 20970,
    "FAUX LEATHER BOMBER JACKET": 71970,
    "FAUX LEATHER BOXY FIT JACKET": 32730,
    "FAUX LEATHER JACKET": 71130,
    "FAUX LEATHER OVERSIZED JACKET LIMITED EDITION": 34910,
    "FAUX LEATHER PUFFER JACKET": 64010,
    "FAUX SHEARLING LINED SUEDE BOOTS": 20760,
    "FAUX SHEARLING PLAID JACKET": 12490,
    "FAUX SUEDE BOMBER JACKET": 50300,
    "FAUX SUEDE JACKET": 24570,
    "FAUX SUEDE OVERSHIRT": 24430,
    "FAUX SUEDE PATCH JACKET": 7070
  }
}