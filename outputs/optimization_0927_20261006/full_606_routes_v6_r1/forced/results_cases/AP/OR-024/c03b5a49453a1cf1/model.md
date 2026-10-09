##### Objective Function:

$\quad \max \sum_{i \in \mathcal{P}} r_i x_i$

where:
- $\mathcal{P}$ is the set of products with identifiers starting with 'S700_'
- $r_i$ is the revenue per unit for product $i$
- $x_i$ is the number of units of product $i$ to fulfill

##### Constraints:

For each $i \in \mathcal{P}$:

$0 \leq x_i \leq \min\{\text{Initial Inventory}_i, \text{Demand}_i\}$

or equivalently, for all $i$:

$0 \leq x_i \leq \text{Initial Inventory}_i$

$0 \leq x_i \leq \text{Demand}_i$

##### Variable Constraints:

$x_i$ is a continuous variable (or integer, if only whole units can be fulfilled), for all $i \in \mathcal{P}$.

##### Retrieved Information

{
  "products": [
    {
      "Product Identifier": "S700_1138",
      "Revenue": 70.67,
      "Initial Inventory": 9020,
      "Demand": 1219
    },
    {
      "Product Identifier": "S700_1691",
      "Revenue": 100.0,
      "Initial Inventory": 8370,
      "Demand": 1127
    },
    {
      "Product Identifier": "S700_1938",
      "Revenue": 70.15,
      "Initial Inventory": 8390,
      "Demand": 1129
    },
    {
      "Product Identifier": "S700_2047",
      "Revenue": 100.0,
      "Initial Inventory": 8680,
      "Demand": 1176
    },
    {
      "Product Identifier": "S700_2466",
      "Revenue": 100.0,
      "Initial Inventory": 9400,
      "Demand": 1301
    },
    {
      "Product Identifier": "S700_2610",
      "Revenue": 65.77,
      "Initial Inventory": 9900,
      "Demand": 1340
    },
    {
      "Product Identifier": "S700_2824",
      "Revenue": 100.0,
      "Initial Inventory": 9760,
      "Demand": 1357
    },
    {
      "Product Identifier": "S700_2834",
      "Revenue": 100.0,
      "Initial Inventory": 8610,
      "Demand": 1158
    },
    {
      "Product Identifier": "S700_3167",
      "Revenue": 74.4,
      "Initial Inventory": 9380,
      "Demand": 1287
    },
    {
      "Product Identifier": "S700_3505",
      "Revenue": 81.14,
      "Initial Inventory": 9170,
      "Demand": 1281
    },
    {
      "Product Identifier": "S700_3962",
      "Revenue": 100.0,
      "Initial Inventory": 8520,
      "Demand": 1135
    },
    {
      "Product Identifier": "S700_4002",
      "Revenue": 61.44,
      "Initial Inventory": 10290,
      "Demand": 1392
    }
  ]
}

##### Parameter Table

| Product Identifier | Revenue ($r_i$) | Initial Inventory | Demand |
|--------------------|-----------------|------------------|--------|
| S700_1138          | 70.67           | 9020             | 1219   |
| S700_1691          | 100.0           | 8370             | 1127   |
| S700_1938          | 70.15           | 8390             | 1129   |
| S700_2047          | 100.0           | 8680             | 1176   |
| S700_2466          | 100.0           | 9400             | 1301   |
| S700_2610          | 65.77           | 9900             | 1340   |
| S700_2824          | 100.0           | 9760             | 1357   |
| S700_2834          | 100.0           | 8610             | 1158   |
| S700_3167          | 74.4            | 9380             | 1287   |
| S700_3505          | 81.14           | 9170             | 1281   |
| S700_3962          | 100.0           | 8520             | 1135   |
| S700_4002          | 61.44           | 10290            | 1392   |

##### Decision Variables

For each product $i$ in the above table, $x_i$ is the number of units to fulfill, subject to:

$0 \leq x_i \leq \min\{\text{Initial Inventory}_i, \text{Demand}_i\}$

##### Full Model

$\max \left(70.67\,x_{\text{S700\_1138}} + 100.0\,x_{\text{S700\_1691}} + 70.15\,x_{\text{S700\_1938}} + 100.0\,x_{\text{S700\_2047}} + 100.0\,x_{\text{S700\_2466}} + 65.77\,x_{\text{S700\_2610}} + 100.0\,x_{\text{S700\_2824}} + 100.0\,x_{\text{S700\_2834}} + 74.4\,x_{\text{S700\_3167}} + 81.14\,x_{\text{S700\_3505}} + 100.0\,x_{\text{S700\_3962}} + 61.44\,x_{\text{S700\_4002}}\right)$

subject to, for each $i$:

$0 \leq x_i \leq \text{Initial Inventory}_i$

$0 \leq x_i \leq \text{Demand}_i$

where $i$ ranges over all products listed above.