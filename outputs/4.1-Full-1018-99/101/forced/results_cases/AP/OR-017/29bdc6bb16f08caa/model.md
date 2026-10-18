##### Decision Variables

Let $x_i$ denote the number of units of product $i$ (where $i$ is a SKU classified as 'ZZ') to fulfill.

##### Objective Function

$\max \sum_{i \in \{\text{ZZ2AO}, \text{ZZDW7}, \text{ZZM1A}, \text{ZZNC5}, \text{ZZX6K}\}} \text{Revenue}_i \cdot x_i$

##### Constraints

For each product $i$:

1. Inventory and Demand Constraints:
$$
0 \leq x_i \leq \min\{\text{Initial Inventory}_i, \text{Demand}_i\}
$$

Or, equivalently, for each $i$:
$$
x_i \leq \text{Initial Inventory}_i
$$
$$
x_i \leq \text{Demand}_i
$$
$$
x_i \geq 0
$$

2. Integrality (if only whole units can be fulfilled):
$$
x_i \in \mathbb{Z}_{\geq 0}
$$

##### Retrieved Information

{
  "products": [
    {
      "SKU": "ZZ2AO",
      "Revenue": 24.38,
      "Demand": 2,
      "Initial Inventory": 10.0
    },
    {
      "SKU": "ZZDW7",
      "Revenue": 30.12,
      "Demand": 4,
      "Initial Inventory": 20.0
    },
    {
      "SKU": "ZZM1A",
      "Revenue": 19.52,
      "Demand": 82,
      "Initial Inventory": 530.0
    },
    {
      "SKU": "ZZNC5",
      "Revenue": 10.79,
      "Demand": 2,
      "Initial Inventory": 10.0
    },
    {
      "SKU": "ZZX6K",
      "Revenue": 111.81,
      "Demand": 2,
      "Initial Inventory": 10.0
    }
  ]
}

##### Parameter Table

| SKU     | Revenue | Demand | Initial Inventory |
|---------|---------|--------|------------------|
| ZZ2AO   | 24.38   | 2      | 10.0             |
| ZZDW7   | 30.12   | 4      | 20.0             |
| ZZM1A   | 19.52   | 82     | 530.0            |
| ZZNC5   | 10.79   | 2      | 10.0             |
| ZZX6K   | 111.81  | 2      | 10.0             |