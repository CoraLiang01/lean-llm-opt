##### Objective Function: 

$ \quad \quad \min \sum_{i=1}^{10} \sum_{t=1}^{4} \bigl( S_{i} \cdot s_{i,t} + C_{i} \cdot w_{i,t} \bigr) $

##### Constraints

###### 1. Startup and activation logic  

$ \quad \quad y_{i,1} = s_{i,1} \quad \forall i=1,\dots,10 $  

$ \quad \quad s_{i,t} \le 1 - y_{i,t-1} \quad \forall i=1,\dots,10,\; t=2,3,4 $  

$ \quad \quad s_{i,4} = 0 \quad \forall i=1,\dots,10 $  

$ \quad \quad y_{i,t} \ge y_{i,t-1} - d_{i,t-1} \quad \forall i=1,\dots,10,\; t=2,3,4 $  

$ \quad \quad y_{i,t+1} \le 1 - d_{i,t} \quad \forall i=1,\dots,10,\; t=1,2 $  

$ \quad \quad y_{i,t+2} \le 1 - d_{i,t} \quad \forall i=1,\dots,10,\; t=1,2 $  

$ \quad \quad s_{i,t+1} \le 1 - d_{i,t} \quad \forall i=1,\dots,10,\; t=1,2 $  

$ \quad \quad d_{i,t} \ge y_{i,t} - y_{i,t+1} \quad \forall i=1,\dots,10,\; t=1,2,3 $  

$ \quad \quad d_{i,t} \le y_{i,t} \quad \forall i=1,\dots,10,\; t=1,2,3 $  

$ \quad \quad d_{i,t} \le 1 - y_{i,t+1} \quad \forall i=1,\dots,10,\; t=1,2,3 $  

###### 2. Capacity and weight limits  

$ \quad \quad w_{i,t} \le Q_{i} \cdot y_{i,t} \quad \forall i=1,\dots,10,\; t=1,\dots,4 $  

$ \quad \quad w_{i,t} \ge 0 \quad \forall i=1,\dots,10,\; t=1,\dots,4 $  

###### 3. Weight change limits  

$ \quad \quad w_{i,1} \le 300 \quad \forall i=1,\dots,10 $  

$ \quad \quad w_{i,t} - w_{i,t-1} \le 300 \quad \forall i=1,\dots,10,\; t=2,3,4 $  

$ \quad \quad w_{i,t-1} - w_{i,t} \le 300 \quad \forall i=1,\dots,10,\; t=2,3,4 $  

###### 4. Demand satisfaction  

$ \quad \quad \sum_{i=1}^{10} w_{i,t} \ge d_{t} \quad \forall t=1,\dots,4 $  

###### 5. Spare‑capacity buffer  

$ \quad \quad \sum_{i=1}^{10} w_{i,t} \le 0.9 \sum_{i=1}^{10} Q_{i} \cdot y_{i,t} \quad \forall t=1,\dots,4 $  

###### 6. Variable domains  

$ \quad \quad y_{i,t}, s_{i,t}, d_{i,t} \in \{0,1\} \quad \forall i=1,\dots,10,\; t=1,\dots,4 $  

$ \quad \quad w_{i,t} \in \mathbb{R}_{\ge 0} \quad \forall i=1,\dots,10,\; t=1,\dots,4 $  

##### Retrieved Information

{
  "trucks": [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10"
  ],
  "capacity": {
    "1": 1000,
    "2": 800,
    "3": 1200,
    "4": 600,
    "5": 900,
    "6": 700,
    "7": 1100,
    "8": 500,
    "9": 1000,
    "10": 650
  },
  "startupCost": {
    "1": 500,
    "2": 300,
    "3": 400,
    "4": 250,
    "5": 450,
    "6": 280,
    "7": 420,
    "8": 200,
    "9": 480,
    "10": 260
  },
  "unitCost": {
    "1": 2.0,
    "2": 3.0,
    "3": 2.5,
    "4": 3.0,
    "5": 2.2,
    "6": 2.8,
    "7": 2.4,
    "8": 3.2,
    "9": 2.1,
    "10": 2.9
  },
  "demand": {
    "1": 1500,
    "2": 2000,
    "3": 1800,
    "4": 1000
  }
}