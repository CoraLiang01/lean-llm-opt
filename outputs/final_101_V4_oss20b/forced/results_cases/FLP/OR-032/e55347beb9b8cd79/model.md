##### Decision Variables

$x_i \geq 0$: number of units of product $i$ that will be fulfilled, for each $i \in I$.

##### Objective Function

$\displaystyle \max \sum_{i\in I} r_i\,x_i$

##### Constraints

1. Demand satisfaction: $\displaystyle x_i \leq d_i,\quad \forall i\in I$  
2. Inventory limitation: $\displaystyle x_i \leq \text{inv}_i,\quad \forall i\in I$  
3. Non‑negativity: $\displaystyle x_i \geq 0,\quad \forall i\in I$  

##### Sets and Parameters

$I=\{\text{Books\_15.15},\ \text{Books\_30.3},\ \text{Books\_45.45},\ \text{Books\_60.6},\ \text{Books\_75.75}\}$  

$r = \begin{bmatrix}15.15\\30.3\\45.45\\60.6\\75.75\end{bmatrix}$  

$d = \begin{bmatrix}1980\\3024\\4536\\5601\\7567\end{bmatrix}$  

$\text{inv} = \begin{bmatrix}9920.0\\20160.0\\30000.0\\38360.0\\51450.0\end{bmatrix}$  

###### Retrieved Information

{
  "products": [
    {
      "Product_Name": "Books_15.15",
      "Revenue": 15.15,
      "Demand": 1980,
      "Initial_Inventory": 9920.0
    },
    {
      "Product_Name": "Books_30.3",
      "Revenue": 30.3,
      "Demand": 3024,
      "Initial_Inventory": 20160.0
    },
    {
      "Product_Name": "Books_45.45",
      "Revenue": 45.45,
      "Demand": 4536,
      "Initial_Inventory": 30000.0
    },
    {
      "Product_Name": "Books_60.6",
      "Revenue": 60.6,
      "Demand": 5601,
      "Initial_Inventory": 38360.0
    },
    {
      "Product_Name": "Books_75.75",
      "Revenue": 75.75,
      "Demand": 7567,
      "Initial_Inventory": 51450.0
    }
  ]
}