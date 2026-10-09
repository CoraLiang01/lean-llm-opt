##### Objective Function:

$\max \sum_{i \in I} \text{Revenue}_i \cdot x_i$

##### Constraints

$\quad 0 \leq x_i \leq \min\{\text{Initial Inventory}_i, \text{Demand}_i\} \quad \forall i \in I$

where $I$ is the set of all car models classified under 'FDK57'.

##### Variable Constraints:

$x_i$ is a continuous decision variable representing the quantity of model $i$ to fulfill.

##### Retrieved Information

{
  "models": [
    {
      "Product Name": "FDK57",
      "Revenue": 119.144,
      "Initial Inventory": 200,
      "Demand": 30
    },
    {
      "Product Name": "FDK57",
      "Revenue": 119.144,
      "Initial Inventory": 100,
      "Demand": 40
    },
    {
      "Product Name": "FDK57",
      "Revenue": 120.144,
      "Initial Inventory": 150,
      "Demand": 50
    },
    {
      "Product Name": "FDK57",
      "Revenue": 121.244,
      "Initial Inventory": 200,
      "Demand": 30
    },
    {
      "Product Name": "FDK57",
      "Revenue": 120.544,
      "Initial Inventory": 150,
      "Demand": 10
    }
  ]
}

##### Parameters

Let $I = \{1,2,3,4,5\}$ correspond to the five 'FDK57' car model entries above.

- $\text{Revenue}_1 = 119.144$, $\text{Initial Inventory}_1 = 200$, $\text{Demand}_1 = 30$
- $\text{Revenue}_2 = 119.144$, $\text{Initial Inventory}_2 = 100$, $\text{Demand}_2 = 40$
- $\text{Revenue}_3 = 120.144$, $\text{Initial Inventory}_3 = 150$, $\text{Demand}_3 = 50$
- $\text{Revenue}_4 = 121.244$, $\text{Initial Inventory}_4 = 200$, $\text{Demand}_4 = 30$
- $\text{Revenue}_5 = 120.544$, $\text{Initial Inventory}_5 = 150$, $\text{Demand}_5 = 10$

##### Explicit Model

$\max \left(119.144\, x_1 + 119.144\, x_2 + 120.144\, x_3 + 121.244\, x_4 + 120.544\, x_5\right)$

subject to

$0 \leq x_1 \leq 30$

$0 \leq x_2 \leq 40$

$0 \leq x_3 \leq 50$

$0 \leq x_4 \leq 30$

$0 \leq x_5 \leq 10$

where $x_i$ is the fulfilled quantity of 'FDK57' car model $i$.

##### Decision Variables

$x_i$: quantity of 'FDK57' car model $i$ to fulfill, for $i=1,\ldots,5$.

##### Notes

- For each model, the upper bound is the minimum of initial inventory and demand.
- All data is retrieved for every 'FDK57' entry in the dataset.