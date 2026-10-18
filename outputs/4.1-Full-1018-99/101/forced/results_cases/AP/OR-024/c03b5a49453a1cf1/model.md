##### Objective Function:

$\quad \max \sum_{i \in \mathcal{P}} r_i x_i$

##### Constraints:

$\quad x_i \leq \text{Inventory}_i \quad \forall i \in \mathcal{P}$

$\quad x_i \leq \text{Demand}_i \quad \forall i \in \mathcal{P}$

$\quad x_i \geq 0 \quad \forall i \in \mathcal{P}$

$\quad x_i \in \mathbb{Z} \quad \forall i \in \mathcal{P}$

##### Retrieved Information

Let $\mathcal{P}$ be the set of products with identifiers starting with 'S700_'. For each $i \in \mathcal{P}$, the parameters are:

| Product ID     | Revenue ($r_i$) | Initial Inventory ($\text{Inventory}_i$) | Demand ($\text{Demand}_i$) |
|----------------|-----------------|------------------------------------------|----------------------------|
| S700_1138      | 70.67           | 9020                                     | 1219                       |
| S700_1691      | 100.0           | 8370                                     | 1127                       |
| S700_1938      | 70.15           | 8390                                     | 1129                       |
| S700_2047      | 100.0           | 8680                                     | 1176                       |
| S700_2466      | 100.0           | 9400                                     | 1301                       |
| S700_2610      | 65.77           | 9900                                     | 1340                       |
| S700_2824      | 100.0           | 9760                                     | 1357                       |
| S700_2834      | 100.0           | 8610                                     | 1158                       |
| S700_3167      | 74.4            | 9380                                     | 1287                       |
| S700_3505      | 81.14           | 9170                                     | 1281                       |
| S700_3962      | 100.0           | 8520                                     | 1135                       |
| S700_4002      | 61.44           | 10290                                    | 1392                       |

##### Variable Definition

For each $i \in \mathcal{P}$, $x_i$ is the number of units of product $i$ to fulfill.

##### Full Model

$\displaystyle \max \left( 70.67\,x_{\text{S700\_1138}} + 100.0\,x_{\text{S700\_1691}} + 70.15\,x_{\text{S700\_1938}} + 100.0\,x_{\text{S700\_2047}} + 100.0\,x_{\text{S700\_2466}} + 65.77\,x_{\text{S700\_2610}} + 100.0\,x_{\text{S700\_2824}} + 100.0\,x_{\text{S700\_2834}} + 74.4\,x_{\text{S700\_3167}} + 81.14\,x_{\text{S700\_3505}} + 100.0\,x_{\text{S700\_3962}} + 61.44\,x_{\text{S700\_4002}} \right)$

Subject to, for each product $i$:

$\quad 0 \leq x_i \leq \min\{\text{Inventory}_i,\, \text{Demand}_i\}$

and $x_i$ integer.

##### Parameter Vectors

- $\mathcal{P} = \{$S700_1138, S700_1691, S700_1938, S700_2047, S700_2466, S700_2610, S700_2824, S700_2834, S700_3167, S700_3505, S700_3962, S700_4002$\}$
- $r = [70.67, 100.0, 70.15, 100.0, 100.0, 65.77, 100.0, 100.0, 74.4, 81.14, 100.0, 61.44]$
- $\text{Inventory} = [9020, 8370, 8390, 8680, 9400, 9900, 9760, 8610, 9380, 9170, 8520, 10290]$
- $\text{Demand} = [1219, 1127, 1129, 1176, 1301, 1340, 1357, 1158, 1287, 1281, 1135, 1392]$