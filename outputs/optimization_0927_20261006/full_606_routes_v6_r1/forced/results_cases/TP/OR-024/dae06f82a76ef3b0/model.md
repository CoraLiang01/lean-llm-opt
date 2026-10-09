##### Decision Variables

For each product $i$ with identifier as below, let $x_i \geq 0$ be the number of units of product $i$ to fulfill (continuous or integer, as appropriate).

##### Parameters

| Product Name   | Revenue ($r_i$) | Demand ($d_i$) | Initial Inventory ($s_i$) |
|:--------------:|:---------------:|:--------------:|:------------------------:|
| S700_1138      | 70.67           | 1219           | 9020                     |
| S700_1691      | 100.0           | 1127           | 8370                     |
| S700_1938      | 70.15           | 1129           | 8390                     |
| S700_2047      | 100.0           | 1176           | 8680                     |
| S700_2466      | 100.0           | 1301           | 9400                     |
| S700_2610      | 65.77           | 1340           | 9900                     |
| S700_2824      | 100.0           | 1357           | 9760                     |
| S700_2834      | 100.0           | 1158           | 8610                     |
| S700_3167      | 74.4            | 1287           | 9380                     |
| S700_3505      | 81.14           | 1281           | 9170                     |
| S700_3962      | 100.0           | 1135           | 8520                     |
| S700_4002      | 61.44           | 1392           | 10290                    |

##### Objective Function

$\max \sum_{i} r_i x_i$

where $r_i$ is the revenue per unit for product $i$.

##### Constraints

For each product $i$:

1. Inventory constraint: $x_i \leq s_i$
2. Demand constraint:  $x_i \leq d_i$
3. Non-negativity:    $x_i \geq 0$

where $s_i$ is the initial inventory and $d_i$ is the demand for product $i$.

##### Complete Model

Let $I$ be the set of products:

$I = \{\text{S700\_1138}, \text{S700\_1691}, \text{S700\_1938}, \text{S700\_2047}, \text{S700\_2466}, \text{S700\_2610}, \text{S700\_2824}, \text{S700\_2834}, \text{S700\_3167}, \text{S700\_3505}, \text{S700\_3962}, \text{S700\_4002}\}$

$\max \left(70.67\,x_{\text{S700\_1138}} + 100.0\,x_{\text{S700\_1691}} + 70.15\,x_{\text{S700\_1938}} + 100.0\,x_{\text{S700\_2047}} + 100.0\,x_{\text{S700\_2466}} + 65.77\,x_{\text{S700\_2610}} + 100.0\,x_{\text{S700\_2824}} + 100.0\,x_{\text{S700\_2834}} + 74.4\,x_{\text{S700\_3167}} + 81.14\,x_{\text{S700\_3505}} + 100.0\,x_{\text{S700\_3962}} + 61.44\,x_{\text{S700\_4002}}\right)$

Subject to, for each $i \in I$:

$x_i \leq s_i$

$x_i \leq d_i$

$x_i \geq 0$

##### Retrieved Information

{
  "S700_1138": {"Revenue": 70.67, "Demand": 1219, "Initial Inventory": 9020},
  "S700_1691": {"Revenue": 100.0, "Demand": 1127, "Initial Inventory": 8370},
  "S700_1938": {"Revenue": 70.15, "Demand": 1129, "Initial Inventory": 8390},
  "S700_2047": {"Revenue": 100.0, "Demand": 1176, "Initial Inventory": 8680},
  "S700_2466": {"Revenue": 100.0, "Demand": 1301, "Initial Inventory": 9400},
  "S700_2610": {"Revenue": 65.77, "Demand": 1340, "Initial Inventory": 9900},
  "S700_2824": {"Revenue": 100.0, "Demand": 1357, "Initial Inventory": 9760},
  "S700_2834": {"Revenue": 100.0, "Demand": 1158, "Initial Inventory": 8610},
  "S700_3167": {"Revenue": 74.4, "Demand": 1287, "Initial Inventory": 9380},
  "S700_3505": {"Revenue": 81.14, "Demand": 1281, "Initial Inventory": 9170},
  "S700_3962": {"Revenue": 100.0, "Demand": 1135, "Initial Inventory": 8520},
  "S700_4002": {"Revenue": 61.44, "Demand": 1392, "Initial Inventory": 10290}
}