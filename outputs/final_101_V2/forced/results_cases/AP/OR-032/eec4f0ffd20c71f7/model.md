##### Decision Variables:

Let $x_i$ denote the number of units of product $i$ (where $i$ indexes the set of ‘Books’ products) to be fulfilled.

##### Objective Function:

$\quad \max \sum_{i=1}^5 r_i x_i$

where $r_i$ is the revenue per unit for product $i$.

##### Constraints:

1. Inventory Constraints:

$\quad x_i \leq s_i \quad \forall i \in \{1,2,3,4,5\}$

where $s_i$ is the initial inventory for product $i$.

2. Demand Constraints:

$\quad x_i \leq d_i \quad \forall i \in \{1,2,3,4,5\}$

where $d_i$ is the demand for product $i$.

3. Non-negativity and Integrality:

$\quad x_i \geq 0,\ x_i \in \mathbb{Z} \quad \forall i \in \{1,2,3,4,5\}$

##### Retrieved Information

{
  "products": [
    "Books_15.15",
    "Books_30.3",
    "Books_45.45",
    "Books_60.6",
    "Books_75.75"
  ],
  "revenue": {
    "Books_15.15": 15.15,
    "Books_30.3": 30.3,
    "Books_45.45": 45.45,
    "Books_60.6": 60.6,
    "Books_75.75": 75.75
  },
  "initial_inventory": {
    "Books_15.15": 9920.0,
    "Books_30.3": 20160.0,
    "Books_45.45": 30000.0,
    "Books_60.6": 38360.0,
    "Books_75.75": 51450.0
  },
  "demand": {
    "Books_15.15": 1980,
    "Books_30.3": 3024,
    "Books_45.45": 4536,
    "Books_60.6": 5601,
    "Books_75.75": 7567
  }
}

##### Full Model (with explicit parameters):

Let the products be indexed as follows:

1: Books_15.15  
2: Books_30.3  
3: Books_45.45  
4: Books_60.6  
5: Books_75.75  

Let $x_1, x_2, x_3, x_4, x_5$ be the fulfilled units for each product.

Objective:

$\max \left(15.15\,x_1 + 30.3\,x_2 + 45.45\,x_3 + 60.6\,x_4 + 75.75\,x_5\right)$

Subject to:

$x_1 \leq 9920.0$  
$x_2 \leq 20160.0$  
$x_3 \leq 30000.0$  
$x_4 \leq 38360.0$  
$x_5 \leq 51450.0$  

$x_1 \leq 1980$  
$x_2 \leq 3024$  
$x_3 \leq 4536$  
$x_4 \leq 5601$  
$x_5 \leq 7567$  

$x_i \geq 0,\ x_i \in \mathbb{Z},\ \forall i \in \{1,2,3,4,5\}$