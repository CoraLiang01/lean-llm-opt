##### Objective Function: 

$\quad \quad \max \bigl(5.5(x_{I,Red}+x_{II,Red}+x_{III,Red})+5(x_{I,Yellow}+x_{II,Yellow}+x_{III,Yellow})+4.8(x_{I,Blue}+x_{II,Blue}+x_{III,Blue})-6(x_{I,Red}+x_{I,Yellow}+x_{I,Blue})-4.5(x_{II,Red}+x_{II,Yellow}+x_{II,Blue})-3(x_{III,Red}+x_{III,Yellow}+x_{III,Blue})\bigr)$

##### Constraints

###### 1. Blending Requirements Constraints

$\quad \quad 9\,x_{I,Red}\;\leq\;x_{II,Red}+x_{III,Red}$  

$\quad \quad x_{II,Red}\;\geq\;x_{I,Red}+x_{III,Red}$  

$\quad \quad 3\,x_{III,Yellow}\;\leq\;7\,x_{I,Yellow}+7\,x_{II,Yellow}$  

$\quad \quad 4\,x_{I,Yellow}\;\geq\;x_{II,Yellow}+x_{III,Yellow}$  

$\quad \quad x_{I,Blue}\;\leq\;x_{II,Blue}+x_{III,Blue}$  

$\quad \quad 9\,x_{II,Blue}\;\geq\;x_{I,Blue}+x_{III,Blue}$  

###### 2. Raw Material Supply Constraints

$\quad \quad x_{I,Red}+x_{I,Yellow}+x_{I,Blue}\;\leq\;1500$  

$\quad \quad x_{II,Red}+x_{II,Yellow}+x_{II,Blue}\;\leq\;2000$  

$\quad \quad x_{III,Red}+x_{III,Yellow}+x_{III,Blue}\;\leq\;1000$  

###### 3. Minimum Production Constraint

$\quad \quad x_{I,Red}+x_{II,Red}+x_{III,Red}\;\geq\;2000$  

###### 4. Non‑negativity and Integrality Constraints

$\quad \quad x_{g,b}\in\mathbb{Z}_{\geq 0}\quad\forall g\in\{I,II,III\},\;b\in\{Red,Yellow,Blue\}$  

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