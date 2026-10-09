##### Decision Variables

$x_i \geq 0$: number of units of ‘Baby’ product $i$ to fulfill (integer).

##### Parameters

- Product set: $I = \{\text{Baby Food\_255.28}\}$
- Revenue per unit: $r_i$  
  $r_{\text{Baby Food\_255.28}} = 255.28$
- Initial Inventory: $s_i$  
  $s_{\text{Baby Food\_255.28}} = 5,\!627,\!060$
- Demand: $d_i$  
  $d_{\text{Baby Food\_255.28}} = 765,\!850$

##### Objective Function

$\max \sum_{i \in I} r_i x_i$

##### Constraints

1. Inventory and demand limits:  
   $0 \leq x_i \leq \min\{s_i, d_i\}, \quad \forall i \in I$

##### Explicit Model

Let $x_{\text{Baby Food\_255.28}}$ be the number of units of Baby Food\_255.28 fulfilled.

$\max\ 255.28\, x_{\text{Baby Food\_255.28}}$

subject to

$0 \leq x_{\text{Baby Food\_255.28}} \leq 765,\!850$

$x_{\text{Baby Food\_255.28}}$ integer.