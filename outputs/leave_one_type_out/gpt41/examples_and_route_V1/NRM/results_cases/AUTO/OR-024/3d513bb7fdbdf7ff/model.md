Let $x_i$ denote the number of units of product $i$ (with Product Name as below) to fulfill.

Objective:
$$
\max \; 70.67\,x_{\text{S700\_1138}} + 100.0\,x_{\text{S700\_1691}} + 70.15\,x_{\text{S700\_1938}} + 100.0\,x_{\text{S700\_2047}} + 100.0\,x_{\text{S700\_2466}} + 65.77\,x_{\text{S700\_2610}} + 100.0\,x_{\text{S700\_2824}} + 100.0\,x_{\text{S700\_2834}} + 74.4\,x_{\text{S700\_3167}} + 81.14\,x_{\text{S700\_3505}} + 100.0\,x_{\text{S700\_3962}} + 61.44\,x_{\text{S700\_4002}}
$$

Subject to, for each product $i$:

- Inventory constraint:
  $$
  x_i \leq \text{Initial Inventory}_i
  $$
- Demand constraint:
  $$
  x_i \leq \text{Demand}_i
  $$
- Nonnegativity and integrality:
  $$
  x_i \in \mathbb{Z}_{\geq 0}
  $$

Where:

| Product Name   | Revenue | Initial Inventory | Demand |
|:--------------|--------:|-----------------:|-------:|
| S700_1138     | 70.67   | 9020             | 1219   |
| S700_1691     | 100.0   | 8370             | 1127   |
| S700_1938     | 70.15   | 8390             | 1129   |
| S700_2047     | 100.0   | 8680             | 1176   |
| S700_2466     | 100.0   | 9400             | 1301   |
| S700_2610     | 65.77   | 9900             | 1340   |
| S700_2824     | 100.0   | 9760             | 1357   |
| S700_2834     | 100.0   | 8610             | 1158   |
| S700_3167     | 74.4    | 9380             | 1287   |
| S700_3505     | 81.14   | 9170             | 1281   |
| S700_3962     | 100.0   | 8520             | 1135   |
| S700_4002     | 61.44   | 10290            | 1392   |

Explicitly, for each $i$:

- $0 \leq x_{\text{S700\_1138}} \leq \min(9020, 1219)$
- $0 \leq x_{\text{S700\_1691}} \leq \min(8370, 1127)$
- $0 \leq x_{\text{S700\_1938}} \leq \min(8390, 1129)$
- $0 \leq x_{\text{S700\_2047}} \leq \min(8680, 1176)$
- $0 \leq x_{\text{S700\_2466}} \leq \min(9400, 1301)$
- $0 \leq x_{\text{S700\_2610}} \leq \min(9900, 1340)$
- $0 \leq x_{\text{S700\_2824}} \leq \min(9760, 1357)$
- $0 \leq x_{\text{S700\_2834}} \leq \min(8610, 1158)$
- $0 \leq x_{\text{S700\_3167}} \leq \min(9380, 1287)$
- $0 \leq x_{\text{S700\_3505}} \leq \min(9170, 1281)$
- $0 \leq x_{\text{S700\_3962}} \leq \min(8520, 1135)$
- $0 \leq x_{\text{S700\_4002}} \leq \min(10290, 1392)$

All $x_i$ are nonnegative integers.