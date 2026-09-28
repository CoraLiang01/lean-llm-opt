##### Decision Variables

$x_i \geq 0$ (integer): Number of units of product $i$ (where $i$ is a ‘27in’ product) to fulfill.

##### Parameters

- Let $I$ be the set of ‘27in’ products:
  - $I = \{\text{27in 4K Gaming Monitor},\ \text{27in FHD Monitor}\}$
- For each $i \in I$:
  - $r_i$: Revenue per unit of product $i$
  - $d_i$: Demand for product $i$
  - $s_i$: Initial Inventory of product $i$

From the retrieved data:
- $r_{\text{27in 4K Gaming Monitor}} = 261.2933$
- $d_{\text{27in 4K Gaming Monitor}} = 12474$
- $s_{\text{27in 4K Gaming Monitor}} = 62440$
- $r_{\text{27in FHD Monitor}} = 52.4965$
- $d_{\text{27in FHD Monitor}} = 15057$
- $s_{\text{27in FHD Monitor}} = 75500$

##### Objective Function

$\max\ 261.2933\,x_{\text{27in 4K Gaming Monitor}} + 52.4965\,x_{\text{27in FHD Monitor}}$

##### Constraints

1. Demand fulfillment (cannot exceed demand):
   - $x_{\text{27in 4K Gaming Monitor}} \leq 12474$
   - $x_{\text{27in FHD Monitor}} \leq 15057$
2. Inventory limit (cannot exceed initial inventory):
   - $x_{\text{27in 4K Gaming Monitor}} \leq 62440$
   - $x_{\text{27in FHD Monitor}} \leq 75500$
3. Non-negativity and integrality:
   - $x_{\text{27in 4K Gaming Monitor}} \geq 0$, integer
   - $x_{\text{27in FHD Monitor}} \geq 0$, integer

##### Complete Model

Let $x_1 = x_{\text{27in 4K Gaming Monitor}}$, $x_2 = x_{\text{27in FHD Monitor}}$.

$\max\ 261.2933\,x_1 + 52.4965\,x_2$

Subject to:
- $x_1 \leq 12474$
- $x_1 \leq 62440$
- $x_2 \leq 15057$
- $x_2 \leq 75500$
- $x_1, x_2 \geq 0$, integer