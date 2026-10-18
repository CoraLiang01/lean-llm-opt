##### Decision Variables

$x_i \geq 0$: number of units of product $i$ fulfilled, for $i \in I$.

##### Objective Function

$\max \sum_{i \in I} r_i x_i$

##### Constraints

1. Demand: $x_i \leq d_i,\quad \forall i \in I$  
2. Inventory: $x_i \leq s_i,\quad \forall i \in I$  
3. Non‑negativity: $x_i \geq 0,\quad \forall i \in I$

$I = \{\text{sku\_I27},\text{sku\_I499},\text{sku\_I719},\text{sku\_T18},\text{sku\_T29},\text{sku\_T39},\text{sku\_T499},\text{sku\_T9},\text{sku\_3081},\text{sku\_339},\text{sku\_3799},\text{sku\_439},\text{sku\_539},\text{sku\_61399},\text{sku\_628},\text{sku\_708},\text{sku\_77},\text{sku\_79},\text{sku\_799},\text{sku\_8499},\text{sku\_89},\text{sku\_897},\text{sku\_9699},\text{sku\_bobo}\}$  

$d_i$ and $s_i$ are the demand and initial inventory for product $i$, respectively; $r_i$ is the revenue per unit for product $i$.

###### Retrieved Information

{
  "product_names": [
    "sku_I27",
    "sku_I499",
    "sku_I719",
    "sku_T18",
    "sku_T29",
    "sku_T39",
    "sku_T499",
    "sku_T9",
    "sku_3081",
    "sku_339",
    "sku_3799",
    "sku_439",
    "sku_539",
    "sku_61399",
    "sku_628",
    "sku_708",
    "sku_77",
    "sku_79",
    "sku_799",
    "sku_8499",
    "sku_89",
    "sku_897",
    "sku_9699",
    "sku_bobo"
  ],
  "revenue": [
    238,
    287,
    268,
    318,
    207,
    258,
    249,
    227,
    198,
    254,
    246,
    258,
    268,
    278,
    268,
    298,
    258,
    315,
    264,
    238,
    258,
    268,
    288,
    228
  ],
  "demand": [
    6,
    4,
    16,
    14,
    4,
    32,
    8,
    2,
    10,
    8,
    18,
    2,
    4,
    8,
    2,
    198,
    32,
    18,
    570,
    6,
    26,
    6,
    33,
    33
  ],
  "initial_inventory": [
    30,
    20,
    80,
    70,
    20,
    160,
    40,
    10,
    50,
    40,
    90,
    10,
    20,
    40,
    10,
    990,
    160,
    90,
    2870,
    30,
    130,
    30,
    170,
    170
  ]
}