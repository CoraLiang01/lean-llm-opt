##### Decision Variables

$x_i \in \mathbb{Z}_{\ge 0}$: number of units of product $i$ to fulfill, where $i \in I$.

##### Objective Function

$\max \sum_{i\in I} r_i x_i$

##### Constraints

1. Demand constraint: $x_i \le d_i,\ \forall i\in I$.

2. Inventory constraint: $x_i \le I_i,\ \forall i\in I$.

3. Domain: $x_i \in \mathbb{Z}_{\ge 0}$.

$I=\{4U\_Service\_22,\ 4U\_Service\_36,\ 4U\_Service\_7\}$.

$M=\sum_{i\in I} d_i=11$ (upper bound on total fulfillment, not required but provided for completeness).

$\,r=\begin{bmatrix}56.0\\21.6\\62.5\end{bmatrix},\quad d=\begin{bmatrix}5\\3\\3\end{bmatrix},\quad I=\begin{bmatrix}30\\20\\20\end{bmatrix}$.

###### Retrieved Information

{
  "products": [
    "4U_Service_22",
    "4U_Service_36",
    "4U_Service_7"
  ],
  "revenue": {
    "4U_Service_22": 56.0,
    "4U_Service_36": 21.6,
    "4U_Service_7": 62.5
  },
  "demand": {
    "4U_Service_22": 5,
    "4U_Service_36": 3,
    "4U_Service_7": 3
  },
  "initial_inventory": {
    "4U_Service_22": 30,
    "4U_Service_36": 20,
    "4U_Service_7": 20
  }
}