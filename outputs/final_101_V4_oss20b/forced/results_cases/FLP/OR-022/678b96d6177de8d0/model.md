##### Decision Variables

$x_i \geq 0$: number of units of product $i$ that will be fulfilled (continuous).  
$i \in I$, where $I=\{1,2\}$ corresponds to the two 27in products.

##### Objective Function

$\displaystyle \max \sum_{i\in I} r_i\,x_i$

##### Constraints

1. Demand limitation: $\displaystyle x_i \leq d_i,\quad \forall i\in I$  
2. Inventory limitation: $\displaystyle x_i \leq I_i,\quad \forall i\in I$  
3. Non‑negativity: $\displaystyle x_i \geq 0,\quad \forall i\in I$

##### Parameters

- $I=\{1,2\}$  
- $d=\begin{bmatrix}12474\\15057\end{bmatrix}$  
- $I_i=\begin{bmatrix}62440\\75500\end{bmatrix}$  
- $r=\begin{bmatrix}261.2933\\52.4965\end{bmatrix}$  

##### Retrieved Information

{
  "products": [
    {
      "Product Name": "27in 4K Gaming Monitor",
      "Revenue": 261.2933,
      "Demand": 12474,
      "Initial Inventory": 62440
    },
    {
      "Product Name": "27in FHD Monitor",
      "Revenue": 52.4965,
      "Demand": 15057,
      "Initial Inventory": 75500
    }
  ],
  "d": [12474, 15057],
  "I_i": [62440, 75500],
  "r": [261.2933, 52.4965]
}