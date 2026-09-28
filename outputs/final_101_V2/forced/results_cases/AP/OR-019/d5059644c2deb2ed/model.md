##### Decision Variables:

Let $x_1$ = number of units of "27in 4K Gaming Monitor" to fulfill  
Let $x_2$ = number of units of "27in FHD Monitor" to fulfill

##### Parameters:

- Revenue per unit:
  - $r_1 = 389.99$ (27in 4K Gaming Monitor)
  - $r_2 = 149.99$ (27in FHD Monitor)
- Initial Inventory:
  - $I_1 = 62,\!440$ (27in 4K Gaming Monitor)
  - $I_2 = 75,\!500$ (27in FHD Monitor)
- Demand:
  - $d_1 = 12,\!474$ (27in 4K Gaming Monitor)
  - $d_2 = 15,\!057$ (27in FHD Monitor)

##### Objective Function:

$\max \; 389.99\,x_1 + 149.99\,x_2$

##### Constraints:

$0 \leq x_1 \leq \min\{62,\!440,\;12,\!474\}$

$0 \leq x_2 \leq \min\{75,\!500,\;15,\!057\}$

$x_1, x_2 \in \mathbb{Z}_{\geq 0}$

##### Retrieved Information

{
  "products": [
    {
      "Product Name": "27in 4K Gaming Monitor",
      "Revenue": 389.99,
      "Initial Inventory": 62440,
      "Demand": 12474
    },
    {
      "Product Name": "27in FHD Monitor",
      "Revenue": 149.99,
      "Initial Inventory": 75500,
      "Demand": 15057
    }
  ]
}