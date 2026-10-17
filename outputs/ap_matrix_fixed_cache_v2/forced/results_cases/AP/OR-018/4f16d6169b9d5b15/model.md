##### Objective Function:

$\quad \max \sum_{i \in \text{Baby}} r_i x_i$

where $r_i$ is the revenue per unit for Baby product $i$, and $x_i$ is the number of units of product $i$ fulfilled.

##### Constraints:

For each Baby product $i$:

$\quad 0 \leq x_i \leq \min\{\text{Initial Inventory}_i, \text{Demand}_i\}$

$\quad x_i \in \mathbb{Z}$

##### Retrieved Information

{
  "products": [
    {
      "Product Name": "Baby Food_255.28",
      "Revenue": 255.28,
      "Demand": 3066513,
      "Initial Inventory": 22749210
    }
  ]
}

##### Model Parameters

Let $i$ index the set of Baby products:

- $r_i$ (Revenue per unit): $r_1 = 255.28$
- $\text{Demand}_i$: $\text{Demand}_1 = 3,\!066,\!513$
- $\text{Initial Inventory}_i$: $\text{Initial Inventory}_1 = 22,\!749,\!210$

##### Model (for this dataset):

$\max\ 255.28\, x_1$

subject to:

$0 \leq x_1 \leq 3,\!066,\!513$

$x_1 \in \mathbb{Z}$

where $x_1$ is the number of units of "Baby Food_255.28" to fulfill.