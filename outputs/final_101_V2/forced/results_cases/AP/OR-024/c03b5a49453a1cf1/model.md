##### Objective Function:

$\quad \max \sum_{i=1}^{12} r_i x_i$

where $r_i$ is the revenue per unit for product $i$, and $x_i$ is the number of units of product $i$ to fulfill.

##### Constraints

###### 1. Inventory and Demand Constraints:

$0 \leq x_i \leq \min\{\text{Initial Inventory}_i, \text{Demand}_i\} \quad \forall i \in \{1,2,\ldots,12\}$

###### 2. Variable Constraints:

$x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,2,\ldots,12\}$

##### Retrieved Information

{
  "products": [
    {
      "Product Name": "S700_1138",
      "Revenue": 70.67,
      "Initial Inventory": 9020,
      "Demand": 1219
    },
    {
      "Product Name": "S700_1691",
      "Revenue": 100.0,
      "Initial Inventory": 8370,
      "Demand": 1127
    },
    {
      "Product Name": "S700_1938",
      "Revenue": 70.15,
      "Initial Inventory": 8390,
      "Demand": 1129
    },
    {
      "Product Name": "S700_2047",
      "Revenue": 100.0,
      "Initial Inventory": 8680,
      "Demand": 1176
    },
    {
      "Product Name": "S700_2466",
      "Revenue": 100.0,
      "Initial Inventory": 9400,
      "Demand": 1301
    },
    {
      "Product Name": "S700_2610",
      "Revenue": 65.77,
      "Initial Inventory": 9900,
      "Demand": 1340
    },
    {
      "Product Name": "S700_2824",
      "Revenue": 100.0,
      "Initial Inventory": 9760,
      "Demand": 1357
    },
    {
      "Product Name": "S700_2834",
      "Revenue": 100.0,
      "Initial Inventory": 8610,
      "Demand": 1158
    },
    {
      "Product Name": "S700_3167",
      "Revenue": 74.4,
      "Initial Inventory": 9380,
      "Demand": 1287
    },
    {
      "Product Name": "S700_3505",
      "Revenue": 81.14,
      "Initial Inventory": 9170,
      "Demand": 1281
    },
    {
      "Product Name": "S700_3962",
      "Revenue": 100.0,
      "Initial Inventory": 8520,
      "Demand": 1135
    },
    {
      "Product Name": "S700_4002",
      "Revenue": 61.44,
      "Initial Inventory": 10290,
      "Demand": 1392
    }
  ]
}

##### Explicit Model

Let $i$ index the products in the order above ($i=1$ for S700_1138, $i=2$ for S700_1691, ..., $i=12$ for S700_4002).

$\max \Big[ 70.67\,x_1 + 100.0\,x_2 + 70.15\,x_3 + 100.0\,x_4 + 100.0\,x_5 + 65.77\,x_6 + 100.0\,x_7 + 100.0\,x_8 + 74.4\,x_9 + 81.14\,x_{10} + 100.0\,x_{11} + 61.44\,x_{12} \Big]$

subject to

$0 \leq x_1 \leq 1219$  ($\min\{9020,1219\}$)  
$0 \leq x_2 \leq 1127$  ($\min\{8370,1127\}$)  
$0 \leq x_3 \leq 1129$  ($\min\{8390,1129\}$)  
$0 \leq x_4 \leq 1176$  ($\min\{8680,1176\}$)  
$0 \leq x_5 \leq 1301$  ($\min\{9400,1301\}$)  
$0 \leq x_6 \leq 1340$  ($\min\{9900,1340\}$)  
$0 \leq x_7 \leq 1357$  ($\min\{9760,1357\}$)  
$0 \leq x_8 \leq 1158$  ($\min\{8610,1158\}$)  
$0 \leq x_9 \leq 1287$  ($\min\{9380,1287\}$)  
$0 \leq x_{10} \leq 1281$ ($\min\{9170,1281\}$)  
$0 \leq x_{11} \leq 1135$ ($\min\{8520,1135\}$)  
$0 \leq x_{12} \leq 1392$ ($\min\{10290,1392\}$)  

$x_i \in \mathbb{Z}_{\geq 0}$ for all $i=1,\ldots,12$