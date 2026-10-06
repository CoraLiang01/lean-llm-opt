##### Objective Function:

$\quad \max \quad 389.99\, x_1 + 149.99\, x_2$

##### Constraints:

$\quad x_1 \leq 12474$  
$\quad x_1 \leq 62440$  

$\quad x_2 \leq 15057$  
$\quad x_2 \leq 75500$  

$\quad x_1, x_2 \in \mathbb{Z}_{\geq 0}$

##### Variable Definitions:

- $x_1$: Number of units fulfilled for "27in 4K Gaming Monitor"
- $x_2$: Number of units fulfilled for "27in FHD Monitor"

##### Parameters (retrieved, source order):

{
  "items": [
    "27in 4K Gaming Monitor",
    "27in FHD Monitor"
  ],
  "revenue": {
    "27in 4K Gaming Monitor": 389.99,
    "27in FHD Monitor": 149.99
  },
  "demand": {
    "27in 4K Gaming Monitor": 12474,
    "27in FHD Monitor": 15057
  },
  "inventory": {
    "27in 4K Gaming Monitor": 62440,
    "27in FHD Monitor": 75500
  }
}

##### Complete Model:

Maximize
$$
389.99\, x_1 + 149.99\, x_2
$$

Subject to
$$
x_1 \leq 12474 \\
x_1 \leq 62440 \\
x_2 \leq 15057 \\
x_2 \leq 75500 \\
x_1, x_2 \in \mathbb{Z}_{\geq 0}
$$