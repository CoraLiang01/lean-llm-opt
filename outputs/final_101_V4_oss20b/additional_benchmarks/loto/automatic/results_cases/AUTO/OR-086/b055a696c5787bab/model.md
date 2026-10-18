##### Objective Function: 

$ \quad \quad \max \sum_{b \in \{Red,Yellow,Blue\}} \sum_{g \in \{I,II,III\}} \bigl( \text{SellingPrice}_b - \text{Cost}_g \bigr) \cdot x_{g,b} $

##### Constraints

###### 1. Blending Requirements Constraints

$\quad \quad x_{I,Red} \leq 0.10 \bigl( x_{I,Red} + x_{II,Red} + x_{III,Red} \bigr)$  

$\quad \quad x_{II,Red} \geq 0.50 \bigl( x_{I,Red} + x_{II,Red} + x_{III,Red} \bigr)$  

$\quad \quad x_{III,Yellow} \leq 0.70 \bigl( x_{I,Yellow} + x_{II,Yellow} + x_{III,Yellow} \bigr)$  

$\quad \quad x_{I,Yellow} \geq 0.20 \bigl( x_{I,Yellow} + x_{II,Yellow} + x_{III,Yellow} \bigr)$  

$\quad \quad x_{I,Blue} \leq 0.50 \bigl( x_{I,Blue} + x_{II,Blue} + x_{III,Blue} \bigr)$  

$\quad \quad x_{II,Blue} \geq 0.10 \bigl( x_{I,Blue} + x_{II,Blue} + x_{III,Blue} \bigr)$  

###### 2. Raw Material Supply Constraints

$\quad \quad x_{I,Red} + x_{I,Yellow} + x_{I,Blue} \leq 1500$  

$\quad \quad x_{II,Red} + x_{II,Yellow} + x_{II,Blue} \leq 2000$  

$\quad \quad x_{III,Red} + x_{III,Yellow} + x_{III,Blue} \leq 1000$  

###### 3. Minimum Production Constraint

$\quad \quad x_{I,Red} + x_{II,Red} + x_{III,Red} \geq 2000$  

###### 4. Non‑negativity Constraints

$\quad \quad x_{g,b} \geq 0, \quad \forall g \in \{I,II,III\}, \; \forall b \in \{Red,Yellow,Blue\}$  

##### Retrieved Information

{
  "grades": [
    {
      "Grade": "I",
      "Daily Supply (kg)": 1500,
      "Cost (CNY/kg)": 6
    },
    {
      "Grade": "II",
      "Daily Supply (kg)": 2000,
      "Cost (CNY/kg)": 4.5
    },
    {
      "Grade": "III",
      "Daily Supply (kg)": 1000,
      "Cost (CNY/kg)": 3
    }
  ],
  "brands": [
    {
      "Brand": "Red",
      "Blending Requirements": "I less than 10%  II more than 50%",
      "Selling Price (CNY/kg)": 5.5
    },
    {
      "Brand": "Yellow",
      "Blending Requirements": "III less than 70%  I more than 20%",
      "Selling Price (CNY/kg)": 5
    },
    {
      "Brand": "Blue",
      "Blending Requirements": "I less than 50%  II more than 10%",
      "Selling Price (CNY/kg)": 4.8
    }
  ]
}