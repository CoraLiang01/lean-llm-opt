##### Decision Variables:

Let $x_i$ denote the number of units of product $i$ (where $i$ indexes the set of 'Books' products) to be fulfilled.

##### Objective Function:

$\quad \max \sum_{i \in \mathcal{B}} r_i x_i$

where:
- $\mathcal{B}$ is the set of 'Books' products,
- $r_i$ is the revenue per unit for product $i$.

##### Constraints:

1. Inventory and Demand Constraints:

$\quad 0 \leq x_i \leq \min\{\text{Initial Inventory}_i, \text{Demand}_i\} \quad \forall i \in \mathcal{B}$

2. Integrality Constraints:

$\quad x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \mathcal{B}$

##### Retrieved Information

{
  "Books_Products": [
    {
      "Product_Name": "Books_15.15",
      "Revenue": 15.15,
      "Demand": 1980,
      "Initial Inventory": 9920.0
    },
    {
      "Product_Name": "Books_30.3",
      "Revenue": 30.3,
      "Demand": 3024,
      "Initial Inventory": 20160.0
    },
    {
      "Product_Name": "Books_45.45",
      "Revenue": 45.45,
      "Demand": 4536,
      "Initial Inventory": 30000.0
    },
    {
      "Product_Name": "Books_60.6",
      "Revenue": 60.6,
      "Demand": 5601,
      "Initial Inventory": 38360.0
    },
    {
      "Product_Name": "Books_75.75",
      "Revenue": 75.75,
      "Demand": 7567,
      "Initial Inventory": 51450.0
    }
  ]
}

##### Explicit Model

Let the set of indices for 'Books' products be $i \in \{1,2,3,4,5\}$, corresponding to the following:

- $i=1$: Books_15.15, $r_1 = 15.15$, $\text{Demand}_1 = 1980$, $\text{Initial Inventory}_1 = 9920.0$
- $i=2$: Books_30.3, $r_2 = 30.3$, $\text{Demand}_2 = 3024$, $\text{Initial Inventory}_2 = 20160.0$
- $i=3$: Books_45.45, $r_3 = 45.45$, $\text{Demand}_3 = 4536$, $\text{Initial Inventory}_3 = 30000.0$
- $i=4$: Books_60.6, $r_4 = 60.6$, $\text{Demand}_4 = 5601$, $\text{Initial Inventory}_4 = 38360.0$
- $i=5$: Books_75.75, $r_5 = 75.75$, $\text{Demand}_5 = 7567$, $\text{Initial Inventory}_5 = 51450.0$

The model is:

$\max \left(15.15\,x_1 + 30.3\,x_2 + 45.45\,x_3 + 60.6\,x_4 + 75.75\,x_5\right)$

subject to:

$0 \leq x_1 \leq 1980$  ($x_1 \leq \min\{9920.0, 1980\} = 1980$)

$0 \leq x_2 \leq 3024$  ($x_2 \leq \min\{20160.0, 3024\} = 3024$)

$0 \leq x_3 \leq 4536$  ($x_3 \leq \min\{30000.0, 4536\} = 4536$)

$0 \leq x_4 \leq 5601$  ($x_4 \leq \min\{38360.0, 5601\} = 5601$)

$0 \leq x_5 \leq 7567$  ($x_5 \leq \min\{51450.0, 7567\} = 7567$)

$x_i \in \mathbb{Z}_{\geq 0}$ for $i=1,\ldots,5$