##### Decision Variables

$x_i \geq 0$: number of units of product $i$ to fulfill.

##### Objective Function

$\max \sum_{i\in I} r_i x_i$

##### Constraints

1. Demand limit: $x_i \leq d_i,\ \forall i\in I$  
2. Inventory limit: $x_i \leq I_i,\ \forall i\in I$  
3. Domains: $x_i \geq 0,\ \forall i\in I$

$I=\{1,2,\dots,12\}$, where each index $i$ corresponds to the $i$‑th ELE‑S product listed below.

###### Retrieved Information

{
  "products": [
    "ELE-SMA-10000463",
    "ELE-SMA-10000487",
    "ELE-SMA-10003333",
    "ELE-SMA-10009012",
    "ELE-SMA-10009999",
    "ELE-SMA-10011234",
    "ELE-SMA-10027456",
    "ELE-SMA-10028567",
    "ELE-SPE-10000484",
    "ELE-SPE-10003030",
    "ELE-SPE-10024123",
    "ELE-SPE-10025234"
  ],
  "revenue": [
    4.0,
    14.0,
    14.0,
    4.0,
    4.0,
    4.0,
    14.0,
    14.0,
    2.4,
    2.4,
    2.4,
    2.4
  ],
  "demand": [
    295,
    1002,
    958,
    777,
    271,
    244,
    990,
    1000,
    169,
    155,
    327,
    174
  ],
  "inventory": [
    2000.0,
    7000.0,
    7000.0,
    6000.0,
    2000.0,
    2000.0,
    7000.0,
    7000.0,
    1200.0,
    1200.0,
    2400.0,
    1200.0
  ]
}