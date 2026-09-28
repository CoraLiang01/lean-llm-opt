##### Decision Variables

$x_i \geq 0$ (integer): Number of units of product $i$ (where $i$ is a ‘27in’ product) to fulfill.

##### Parameters

- Products $i \in \{\text{27in 4K Gaming Monitor},\ \text{27in FHD Monitor}\}$
- Revenue per unit $r_i$:
  - $r_{\text{27in 4K Gaming Monitor}} = 389.99$
  - $r_{\text{27in FHD Monitor}} = 149.99$
- Initial inventory $I_i$:
  - $I_{\text{27in 4K Gaming Monitor}} = 62440$
  - $I_{\text{27in FHD Monitor}} = 75500$
- Demand $d_i$:
  - $d_{\text{27in 4K Gaming Monitor}} = 12474$
  - $d_{\text{27in FHD Monitor}} = 15057$

##### Objective Function

$$
\max\ r_{\text{27in 4K Gaming Monitor}}\, x_{\text{27in 4K Gaming Monitor}} + r_{\text{27in FHD Monitor}}\, x_{\text{27in FHD Monitor}}
$$

or numerically,

$$
\max\ 389.99\, x_{\text{27in 4K Gaming Monitor}} + 149.99\, x_{\text{27in FHD Monitor}}
$$

##### Constraints

1. Inventory constraints:
   $$
   x_{\text{27in 4K Gaming Monitor}} \leq 62440
   $$
   $$
   x_{\text{27in FHD Monitor}} \leq 75500
   $$

2. Demand constraints:
   $$
   x_{\text{27in 4K Gaming Monitor}} \leq 12474
   $$
   $$
   x_{\text{27in FHD Monitor}} \leq 15057
   $$

3. Nonnegativity and integrality:
   $$
   x_{\text{27in 4K Gaming Monitor}} \geq 0,\quad x_{\text{27in FHD Monitor}} \geq 0
   $$
   $$
   x_{\text{27in 4K Gaming Monitor}},\ x_{\text{27in FHD Monitor}} \in \mathbb{Z}
   $$

##### Complete Model

Let $i \in \{\text{27in 4K Gaming Monitor},\ \text{27in FHD Monitor}\}$.

$$
\max\ \sum_{i} r_i x_i
$$

subject to

$$
x_i \leq I_i,\quad \forall i
$$

$$
x_i \leq d_i,\quad \forall i
$$

$$
x_i \geq 0,\ x_i \in \mathbb{Z},\quad \forall i
$$

Where:

- $r_{\text{27in 4K Gaming Monitor}} = 389.99$, $I_{\text{27in 4K Gaming Monitor}} = 62440$, $d_{\text{27in 4K Gaming Monitor}} = 12474$
- $r_{\text{27in FHD Monitor}} = 149.99$, $I_{\text{27in FHD Monitor}} = 75500$, $d_{\text{27in FHD Monitor}} = 15057$