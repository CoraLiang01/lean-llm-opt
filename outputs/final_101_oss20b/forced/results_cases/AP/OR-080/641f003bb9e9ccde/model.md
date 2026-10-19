##### Objective Function: 

$\quad \quad \min \sum_{i=1}^{10} \sum_{t=1}^{4} \bigl( S_i y_{i,t} + C_i w_{i,t} \bigr)$ 

##### Constraints

###### 1. Startup Definition Constraints: 

$y_{i,t} \le x_{i,t} \quad \forall i \in \{1,\dots ,10\}, \; t \in \{1,\dots ,4\}$  

$y_{i,t} \le 1 - x_{i,t-1} \quad \forall i \in \{1,\dots ,10\}, \; t \in \{1,\dots ,4\}$  

$x_{i,0} = 0 \quad \forall i \in \{1,\dots ,10\}$  

###### 2. Minimum Two‑Consecutive‑Period Constraints: 

$x_{i,t} \le x_{i,t+1} \quad \forall i \in \{1,\dots ,10\}, \; t \in \{1,2,3\}$  

###### 3. Shutdown Constraints: 

$x_{i,t-1} + x_{i,t+1} \le 1 + x_{i,t} \quad \forall i \in \{1,\dots ,10\}, \; t \in \{2,3\}$  

###### 4. Weight Capacity Constraints: 

$0 \le w_{i,t} \le Q_i x_{i,t} \quad \forall i \in \{1,\dots ,10\}, \; t \in \{1,\dots ,4\}$  

###### 5. Weight Change Constraints: 

$w_{i,t} - w_{i,t-1} \le 300 \quad \forall i \in \{1,\dots ,10\}, \; t \in \{1,\dots ,4\}$  

$w_{i,t-1} - w_{i,t} \le 300 \quad \forall i \in \{1,\dots ,10\}, \; t \in \{1,\dots ,4\}$  

$w_{i,0} = 0 \quad \forall i \in \{1,\dots ,10\}$  

###### 6. Demand Satisfaction Constraints: 

$\sum_{i=1}^{10} w_{i,t} \ge d_t \quad \forall t \in \{1,2,3,4\}$  

###### 7. Spare‑Capacity Constraints: 

$\sum_{i=1}^{10} w_{i,t} \le 0.9 \sum_{i=1}^{10} Q_i x_{i,t} \quad \forall t \in \{1,2,3,4\}$  

###### 8. No Startup in Period 4: 

$y_{i,4} = 0 \quad \forall i \in \{1,\dots ,10\}$  

###### 9. Variable Domain Constraints: 

$x_{i,t}, y_{i,t} \in \{0,1\} \quad \forall i \in \{1,\dots ,10\}, \; t \in \{1,\dots ,4\}$  

###### 10. Weight Variable Domain Constraints: 

$w_{i,t} \ge 0 \quad \forall i \in \{1,\dots ,10\}, \; t \in \{1,\dots ,4\}$  

###### Retrieved Information

{
  "truck_ids": ["1","2","3","4","5","6","7","8","9","10"],
  "Q": {"1":1000,"2":800,"3":1200,"4":600,"5":900,"6":700,"7":1100,"8":500,"9":1000,"10":650},
  "S": {"1":500,"2":300,"3":400,"4":250,"5":450,"6":280,"7":420,"8":200,"9":480,"10":260},
  "C": {"1":2.0,"2":3.0,"3":2.5,"4":3.0,"5":2.2,"6":2.8,"7":2.4,"8":3.2,"9":2.1,"10":2.9},
  "demand": {"1":1500,"2":2000,"3":1800,"4":1000}
}