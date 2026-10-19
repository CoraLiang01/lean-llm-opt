##### Objective Function: 

$ \quad \quad \max \sum_{i \in \{1,\dots ,24\}} r_{i} x_{i} $

##### Constraints

###### 1. Demand Constraints: 

$ x_{i} \leq d_{i} \quad \forall i \in \{1,\dots ,24\} $

###### 2. Inventory Constraints: 

$ x_{i} \leq I_{i} \quad \forall i \in \{1,\dots ,24\} $

###### 3. Non‑negativity Constraints: 

$ x_{i} \geq 0 \quad \forall i \in \{1,\dots ,24\} $

###### 4. Integer Constraints: 

$ x_{i} \in \mathbb{Z} \quad \forall i \in \{1,\dots ,24\} $

###### 5. Variable Domain: 

$ x_{i} \in \{0,1,2,\dots \} \quad \forall i \in \{1,\dots ,24\} $

###### 6. Parameter Definitions: 

$ r_{i} $ – revenue per unit of product $i$  
$ d_{i} $ – demand quantity of product $i$  
$ I_{i} $ – initial inventory of product $i$  

###### 7. Index Mapping: 

$ i=1 \rightarrow \text{sku\_I27}$  
$ i=2 \rightarrow \text{sku\_I499}$  
$ i=3 \rightarrow \text{sku\_I719}$  
$ i=4 \rightarrow \text{sku\_T18}$  
$ i=5 \rightarrow \text{sku\_T29}$  
$ i=6 \rightarrow \text{sku\_T39}$  
$ i=7 \rightarrow \text{sku\_T499}$  
$ i=8 \rightarrow \text{sku\_T9}$  
$ i=9 \rightarrow \text{sku\_3081}$  
$ i=10 \rightarrow \text{sku\_339}$  
$ i=11 \rightarrow \text{sku\_3799}$  
$ i=12 \rightarrow \text{sku\_439}$  
$ i=13 \rightarrow \text{sku\_539}$  
$ i=14 \rightarrow \text{sku\_61399}$  
$ i=15 \rightarrow \text{sku\_628}$  
$ i=16 \rightarrow \text{sku\_708}$  
$ i=17 \rightarrow \text{sku\_77}$  
$ i=18 \rightarrow \text{sku\_79}$  
$ i=19 \rightarrow \text{sku\_799}$  
$ i=20 \rightarrow \text{sku\_8499}$  
$ i=21 \rightarrow \text{sku\_89}$  
$ i=22 \rightarrow \text{sku\_897}$  
$ i=23 \rightarrow \text{sku\_9699}$  
$ i=24 \rightarrow \text{sku\_bobo}$

###### 8. Parameter Values:

$ r = \{238, 287, 268, 318, 207, 258, 249, 227, 198, 254, 246, 258, 268, 278, 268, 298, 258, 315, 264, 238, 258, 268, 288, 228\} $

$ d = \{6, 4, 16, 14, 4, 32, 8, 2, 10, 8, 18, 2, 4, 8, 2, 198, 32, 18, 570, 6, 26, 6, 33, 33\} $

$ I = \{30, 20, 80, 70, 20, 160, 40, 10, 50, 40, 90, 10, 20, 40, 10, 990, 160, 90, 2870, 30, 130, 30, 170, 170\} $

##### Retrieved Information

{
  "products": [
    "sku_I27",
    "sku_I499",
    "sku_I719",
    "sku_T18",
    "sku_T29",
    "sku_T39",
    "sku_T499",
    "sku_T9",
    "sku_3081",
    "sku_339",
    "sku_3799",
    "sku_439",
    "sku_539",
    "sku_61399",
    "sku_628",
    "sku_708",
    "sku_77",
    "sku_79",
    "sku_799",
    "sku_8499",
    "sku_89",
    "sku_897",
    "sku_9699",
    "sku_bobo"
  ],
  "revenue": {
    "sku_I27": 238,
    "sku_I499": 287,
    "sku_I719": 268,
    "sku_T18": 318,
    "sku_T29": 207,
    "sku_T39": 258,
    "sku_T499": 249,
    "sku_T9": 227,
    "sku_3081": 198,
    "sku_339": 254,
    "sku_3799": 246,
    "sku_439": 258,
    "sku_539": 268,
    "sku_61399": 278,
    "sku_628": 268,
    "sku_708": 298,
    "sku_77": 258,
    "sku_79": 315,
    "sku_799": 264,
    "sku_8499": 238,
    "sku_89": 258,
    "sku_897": 268,
    "sku_9699": 288,
    "sku_bobo": 228
  },
  "demand": {
    "sku_I27": 6,
    "sku_I499": 4,
    "sku_I719": 16,
    "sku_T18": 14,
    "sku_T29": 4,
    "sku_T39": 32,
    "sku_T499": 8,
    "sku_T9": 2,
    "sku_3081": 10,
    "sku_339": 8,
    "sku_3799": 18,
    "sku_439": 2,
    "sku_539": 4,
    "sku_61399": 8,
    "sku_628": 2,
    "sku_708": 198,
    "sku_77": 32,
    "sku_79": 18,
    "sku_799": 570,
    "sku_8499": 6,
    "sku_89": 26,
    "sku_897": 6,
    "sku_9699": 33,
    "sku_bobo": 33
  },
  "initial_inventory": {
    "sku_I27": 30,
    "sku_I499": 20,
    "sku_I719": 80,
    "sku_T18": 70,
    "sku_T29": 20,
    "sku_T39": 160,
    "sku_T499": 40,
    "sku_T9": 10,
    "sku_3081": 50,
    "sku_339": 40,
    "sku_3799": 90,
    "sku_439": 10,
    "sku_539": 20,
    "sku_61399": 40,
    "sku_628": 10,
    "sku_708": 990,
    "sku_77": 160,
    "sku_79": 90,
    "sku_799": 2870,
    "sku_8499": 30,
    "sku_89": 130,
    "sku_897": 30,
    "sku_9699": 170,
    "sku_bobo": 170
  }
}