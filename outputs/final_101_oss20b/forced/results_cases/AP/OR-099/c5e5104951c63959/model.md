##### Objective Function: 

$ \quad \quad \min \sum_{i=1}^{11} f_i y_i + \sum_{i=1}^{11}\sum_{j=1}^{11} c_{ij} x_{ij} $

##### Constraints

###### 1. Demand Satisfaction Constraints: 

$ \sum_{i=1}^{11} x_{ij} = d_j \quad \forall j \in \{1,\dots ,11\} $

###### 2. Capacity Constraints: 

$ \sum_{j=1}^{11} x_{ij} \le K_i y_i \quad \forall i \in \{1,\dots ,11\} $

###### 3. Variable Constraints: 

$ x_{ij} \ge 0 \quad \forall i,j $  

$ y_i \in \{0,1\} \quad \forall i $

###### 4. Binary Decision Variables: 

$ y_i \in \{0,1\} \quad \forall i $

###### 5. Flow Variables: 

$ x_{ij} \ge 0 \quad \forall i,j $

###### 6. Opening Cost Variables: 

$ f_i \ge 0 \quad \forall i $

###### 7. Capacity Variables: 

$ K_i \ge 0 \quad \forall i $

###### 8. Demand Variables: 

$ d_j \ge 0 \quad \forall j $

###### 9. Transportation Cost Variables: 

$ c_{ij} \ge 0 \quad \forall i,j $

###### 10. Warehouse Identification: 

$ i \in \{1,\dots ,11\} $

###### 11. Store Identification: 

$ j \in \{1,\dots ,11\} $

###### 12. Non-negativity of Flow: 

$ x_{ij} \ge 0 \quad \forall i,j $

###### 13. Binary Opening Decision: 

$ y_i \in \{0,1\} \quad \forall i $

###### 14. Capacity Binding: 

$ \sum_{j=1}^{11} x_{ij} \le K_i y_i \quad \forall i $

###### 15. Demand Fulfillment: 

$ \sum_{i=1}^{11} x_{ij} = d_j \quad \forall j $

###### 16. Variable Domains: 

$ x_{ij} \in \mathbb{R}_{+}, \; y_i \in \{0,1\} \quad \forall i,j $

###### 17. Parameter Definitions: 

$ f_i $ – opening cost of warehouse $i$  
$ K_i $ – capacity of warehouse $i$  
$ d_j $ – demand of store $j$  
$ c_{ij} $ – transportation cost from warehouse $i$ to store $j$  

###### 18. Index Sets: 

$ I = \{1,\dots ,11\} $ – set of warehouses  
$ J = \{1,\dots ,11\} $ – set of stores  

###### 19. Decision Variables: 

$ y_i $ – binary variable indicating if warehouse $i$ is opened  
$ x_{ij} $ – amount of goods shipped from warehouse $i$ to store $j$  

###### 20. Objective Coefficients: 

$ f_i $ – fixed opening cost for warehouse $i$  
$ c_{ij} $ – variable transportation cost per unit from warehouse $i$ to store $j$  

###### 21. Capacity Constraints: 

$ \sum_{j \in J} x_{ij} \le K_i y_i \quad \forall i \in I $  

###### 22. Demand Constraints: 

$ \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J $  

###### 23. Variable Bounds: 

$ x_{ij} \ge 0 \quad \forall i \in I, j \in J $  
$ y_i \in \{0,1\} \quad \forall i \in I $  

###### 24. Parameter Values: 

$ f_i = \{3000,3200,3100,2800,3500,2700,2900,3050,3100,2200,2890\} $  
$ K_i = \{180,160,200,150,170,190,160,175,170,180,190\} $  
$ d_j = \{30,40,20,35,20,25,45,38,32,41,44\} $  
$ c_{ij} = \begin{bmatrix}
12 & 11 & 14 & 15 & 17 & 13 & 12 & 16 & 16 & 14 & 15\\
17 & 19 & 15 & 20 & 18 & 14 & 17 & 15 & 13 & 15 & 16\\
13 & 14 & 12 & 14 & 16 & 15 & 11 & 14 & 16 & 18 & 17\\
18 & 16 & 17 & 13 & 18 & 17 & 14 & 19 & 16 & 13 & 18\\
10 & 13 & 12 & 19 & 15 & 11 & 12 & 14 & 12 & 15 & 17\\
15 & 12 & 14 & 16 & 13 & 17 & 16 & 16 & 14 & 18 & 19\\
14 & 13 & 15 & 17 & 12 & 13 & 14 & 15 & 12 & 16 & 14\\
19 & 16 & 18 & 20 & 17 & 19 & 16 & 18 & 15 & 15 & 18\\
17 & 18 & 12 & 14 & 16 & 15 & 14 & 17 & 21 & 15 & 18\\
14 & 13 & 15 & 17 & 16 & 18 & 14 & 19 & 15 & 17 & 19\\
15 & 13 & 16 & 17 & 11 & 13 & 14 & 15 & 19 & 21 & 13
\end{bmatrix}$  

###### 25. Index Mapping: 

$ i = 1 \rightarrow \text{Warehouse 1}, \dots , i = 11 \rightarrow \text{Warehouse 11}$  
$ j = 1 \rightarrow \text{Store 1}, \dots , j = 11 \rightarrow \text{Store 11}$  

###### 26. Summary of Decision Variables: 

$ y_i \in \{0,1\} $ for $i=1,\dots ,11$  
$ x_{ij} \ge 0 $ for $i=1,\dots ,11$, $j=1,\dots ,11$  

###### 27. Final Model: 

$ \min \sum_{i=1}^{11} f_i y_i + \sum_{i=1}^{11}\sum_{j=1}^{11} c_{ij} x_{ij} $  
subject to constraints 1–24 above.  

###### 28. Retrieved Information

{
  "warehouses": [
    {"id":"W1","opening_cost":3000,"capacity":180},
    {"id":"W2","opening_cost":3200,"capacity":160},
    {"id":"W3","opening_cost":3100,"capacity":200},
    {"id":"W4","opening_cost":2800,"capacity":150},
    {"id":"W5","opening_cost":3500,"capacity":170},
    {"id":"W6","opening_cost":2700,"capacity":190},
    {"id":"W7","opening_cost":2900,"capacity":160},
    {"id":"W8","opening_cost":3050,"capacity":175},
    {"id":"W9","opening_cost":3100,"capacity":170},
    {"id":"W10","opening_cost":2200,"capacity":180},
    {"id":"W11","opening_cost":2890,"capacity":190}
  ],
  "stores": [
    {"id":"S1","demand":30},
    {"id":"S2","demand":40},
    {"id":"S3","demand":20},
    {"id":"S4","demand":35},
    {"id":"S5","demand":20},
    {"id":"S6","demand":25},
    {"id":"S7","demand":45},
    {"id":"S8","demand":38},
    {"id":"S9","demand":32},
    {"id":"S10","demand":41},
    {"id":"S11","demand":44}
  ],
  "transportation_costs": {
    "W1":{"S1":12,"S2":11,"S3":14,"S4":15,"S5":17,"S6":13,"S7":12,"S8":16,"S9":16,"S10":14,"S11":15},
    "W2":{"S1":17,"S2":19,"S3":15,"S4":20,"S5":18,"S6":14,"S7":17,"S8":15,"S9":13,"S10":15,"S11":16},
    "W3":{"S1":13,"S2":14,"S3":12,"S4":14,"S5":16,"S6":15,"S7":11,"S8":14,"S9":16,"S10":18,"S11":17},
    "W4":{"S1":18,"S2":16,"S3":17,"S4":13,"S5":18,"S6":17,"S7":14,"S8":19,"S9":16,"S10":13,"S11":18},
    "W5":{"S1":10,"S2":13,"S3":12,"S4":19,"S5":15,"S6":11,"S7":12,"S8":14,"S9":12,"S10":15,"S11":17},
    "W6":{"S1":15,"S2":12,"S3":14,"S4":16,"S5":13,"S6":17,"S7":16,"S8":16,"S9":14,"S10":18,"S11":19},
    "W7":{"S1":14,"S2":13,"S3":15,"S4":17,"S5":12,"S6":13,"S7":14,"S8":15,"S9":12,"S10":16,"S11":14},
    "W8":{"S1":19,"S2":16,"S3":18,"S4":20,"S5":17,"S6":19,"S7":16,"S8":18,"S9":15,"S10":15,"S11":18},
    "W9":{"S1":17,"S2":18,"S3":12,"S4":14,"S5":16,"S6":15,"S7":14,"S8":17,"S9":21,"S10":15,"S11":18},
    "W10":{"S1":14,"S2":13,"S3":15,"S4":17,"S5":16,"S6":18,"S7":14,"S8":19,"S9":15,"S10":17,"S11":19},
    "W11":{"S1":15,"S2":13,"S3":16,"S4":17,"S5":11,"S6":13,"S7":14,"S8":15,"S9":19,"S10":21,"S11":13}
  }
}