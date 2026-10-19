##### Objective Function: 

$$\max \sum_{b \in \{Red,Yellow,Blue\}} \sum_{g \in \{I,II,III\}} p_{b}\,x_{g,b} \;-\; \sum_{g \in \{I,II,III\}} \sum_{b \in \{Red,Yellow,Blue\}} c_{g}\,x_{g,b} $$

##### Constraints

###### 1. Raw Material Supply Constraints: 

$$\sum_{b \in \{Red,Yellow,Blue\}} x_{I,b} \;\le\; 1500$$
$$\sum_{b \in \{Red,Yellow,Blue\}} x_{II,b} \;\le\; 2000$$
$$\sum_{b \in \{Red,Yellow,Blue\}} x_{III,b} \;\le\; 1000$$

###### 2. Minimum Production Constraint: 

$$\sum_{g \in \{I,II,III\}} x_{g,Red} \;\ge\; 2000$$

###### 3. Blending Requirements Constraints: 

**Red Brand**  
$$x_{I,Red} \;\le\; 0.10 \;\sum_{g' \in \{I,II,III\}} x_{g',Red}$$  
$$x_{II,Red} \;\ge\; 0.50 \;\sum_{g' \in \{I,II,III\}} x_{g',Red}$$  

**Yellow Brand**  
$$x_{III,Yellow} \;\le\; 0.70 \;\sum_{g' \in \{I,II,III\}} x_{g',Yellow}$$  
$$x_{I,Yellow} \;\ge\; 0.20 \;\sum_{g' \in \{I,II,III\}} x_{g',Yellow}$$  

**Blue Brand**  
$$x_{I,Blue} \;\le\; 0.50 \;\sum_{g' \in \{I,II,III\}} x_{g',Blue}$$  
$$x_{II,Blue} \;\ge\; 0.10 \;\sum_{g' \in \{I,II,III\}} x_{g',Blue}$$  

###### 4. Non‑negativity Constraints: 

$$x_{g,b} \;\ge\; 0 \quad \forall g \in \{I,II,III\},\; \forall b \in \{Red,Yellow,Blue\}$$

##### Retrieved Information

{
  "cost": {
    "I": 6,
    "II": 4.5,
    "III": 3
  },
  "supply": {
    "I": 1500,
    "II": 2000,
    "III": 1000
  },
  "price": {
    "Red": 5.5,
    "Yellow": 5,
    "Blue": 4.8
  },
  "blending_requirements": {
    "Red": {
      "I_upper": 0.10,
      "II_lower": 0.50
    },
    "Yellow": {
      "III_upper": 0.70,
      "I_lower": 0.20
    },
    "Blue": {
      "I_upper": 0.50,
      "II_lower": 0.10
    }
  }
}