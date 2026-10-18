##### Decision Variables:

Let $x_i$ denote the number of units of Organ product $i$ to be fulfilled, for each $i$ in the set of Organ products:
- $i=1$: Organic Fruits
- $i=2$: Organic Staples
- $i=3$: Organic Vegetables

##### Parameters:

- Revenue per unit:
  - $r_1 = 60.8$ (Organic Fruits)
  - $r_2 = 918.45$ (Organic Staples)
  - $r_3 = 77.52$ (Organic Vegetables)

- Initial Inventory:
  - $I_1 = 5,\!034,\!020.0$ (Organic Fruits)
  - $I_2 = 5,\!589,\!290.0$ (Organic Staples)
  - $I_3 = 5,\!202,\!710.0$ (Organic Vegetables)

- Demand:
  - $d_1 = 678,\!906$ (Organic Fruits)
  - $d_2 = 749,\!927$ (Organic Staples)
  - $d_3 = 699,\!808$ (Organic Vegetables)

##### Objective Function:

$\max \left( 60.8\,x_1 + 918.45\,x_2 + 77.52\,x_3 \right)$

##### Constraints:

For each $i \in \{1,2,3\}$:
- $0 \leq x_i \leq \min\{I_i, d_i\}$

Explicitly:
- $0 \leq x_1 \leq 678,\!906$
- $0 \leq x_2 \leq  749,\!927$
- $0 \leq x_3 \leq 699,\!808$

##### Variable Domains:

$x_i$ are integer variables for all $i \in \{1,2,3\}$

##### Retrieved Information

{
  "products": [
    {
      "Sub Category": "Organic Fruits",
      "Revenue": 60.8,
      "Demand": 678906,
      "Initial Inventory": 5034020.0
    },
    {
      "Sub Category": "Organic Staples",
      "Revenue": 918.45,
      "Demand": 749927,
      "Initial Inventory": 5589290.0
    },
    {
      "Sub Category": "Organic Vegetables",
      "Revenue": 77.52,
      "Demand": 699808,
      "Initial Inventory": 5202710.0
    }
  ]
}