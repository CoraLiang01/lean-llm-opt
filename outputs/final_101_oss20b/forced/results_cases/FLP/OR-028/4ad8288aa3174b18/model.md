##### Decision Variables

$x_i \geq 0$: number of units of product $i$ that will be fulfilled (continuous).

##### Objective Function

$\max \sum_{i} \text{Revenue}_i\,x_i$

##### Constraints

1. Demand satisfaction: $\displaystyle x_i \leq \text{Demand}_i,\quad \forall i$  
2. Inventory limitation: $\displaystyle x_i \leq \text{Initial Inventory}_i,\quad \forall i$  
3. Non‑negativity: $\displaystyle x_i \geq 0,\quad \forall i$

##### Parameters

**Products**  
$P=\{p_1,p_2,\dots ,p_{24}\}$  

| $i$ | Product Name | Revenue$_i$ | Demand$_i$ | Initial Inventory$_i$ |
|-----|--------------|-------------|------------|------------------------|
| 1 | sku_I27 | 238 | 6 | 30 |
| 2 | sku_I499 | 287 | 4 | 20 |
| 3 | sku_I719 | 268 | 16 | 80 |
| 4 | sku_T18 | 318 | 14 | 70 |
| 5 | sku_T29 | 207 | 4 | 20 |
| 6 | sku_T39 | 258 | 32 | 160 |
| 7 | sku_T499 | 249 | 8 | 40 |
| 8 | sku_T9 | 227 | 2 | 10 |
| 9 | sku_3081 | 198 | 10 | 50 |
|10 | sku_339 | 254 | 8 | 40 |
|11 | sku_3799 | 246 | 18 | 90 |
|12 | sku_439 | 258 | 2 | 10 |
|13 | sku_539 | 268 | 4 | 20 |
|14 | sku_61399 | 278 | 8 | 40 |
|15 | sku_628 | 268 | 2 | 10 |
|16 | sku_708 | 298 | 198 | 990 |
|17 | sku_77 | 258 | 32 | 160 |
|18 | sku_79 | 315 | 18 | 90 |
|19 | sku_799 | 264 | 570 | 2870 |
|20 | sku_8499 | 238 | 6 | 30 |
|21 | sku_89 | 258 | 26 | 130 |
|22 | sku_897 | 268 | 6 | 30 |
|23 | sku_9699 | 288 | 33 | 170 |
|24 | sku_bobo | 228 | 33 | 170 |

**Vectors**

$\text{Revenue} = [238,\;287,\;268,\;318,\;207,\;258,\;249,\;227,\;198,\;254,\;246,\;258,\;268,\;278,\;268,\;298,\;258,\;315,\;264,\;238,\;258,\;268,\;288,\;228]$

$\text{Demand} = [6,\;4,\;16,\;14,\;4,\;32,\;8,\;2,\;10,\;8,\;18,\;2,\;4,\;8,\;2,\;198,\;32,\;18,\;570,\;6,\;26,\;6,\;33,\;33]$

$\text{Initial Inventory} = [30,\;20,\;80,\;70,\;20,\;160,\;40,\;10,\;50,\;40,\;90,\;10,\;20,\;40,\;10,\;990,\;160,\;90,\;2870,\;30,\;130,\;30,\;170,\;170]$