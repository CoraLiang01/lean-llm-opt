##### Objective Function: 

$\quad \quad \max \sum_{i=1}^{35} r_{i} x_{i}$

##### Constraints

###### 1. Demand Constraints: 

$\quad \quad 0 \le x_{i} \le d_{i} \quad \forall i \in \{1,\dots ,35\}$

###### 2. Inventory Constraints: 

$\quad \quad 0 \le x_{i} \le I_{i} \quad \forall i \in \{1,\dots ,35\}$

###### 3. Variable Constraints: 

$\quad \quad x_{i} \ge 0 \quad \forall i \in \{1,\dots ,35\}$

###### 4. (Optional) Integer Constraints if units must be whole numbers: 

$\quad \quad x_{i} \in \mathbb{Z}_{+} \quad \forall i \in \{1,\dots ,35\}$

###### 5. (Optional) Total Inventory Constraint if a global inventory limit exists: 

$\quad \quad \sum_{i=1}^{35} x_{i} \le \sum_{i=1}^{35} I_{i}$

##### Retrieved Information

{
  "products": [
    {"name":"Beauty - 25","revenue":25,"demand":240,"inventory":1570},
    {"name":"Beauty - 30","revenue":30,"demand":202,"inventory":1330},
    {"name":"Beauty - 300","revenue":300,"demand":216,"inventory":1420},
    {"name":"Beauty - 50","revenue":50,"demand":263,"inventory":1700},
    {"name":"Beauty - 500","revenue":500,"demand":256,"inventory":1690},
    {"name":"Clothing - 25","revenue":25,"demand":281,"inventory":1840},
    {"name":"Clothing - 30","revenue":30,"demand":261,"inventory":1710},
    {"name":"Clothing - 300","revenue":300,"demand":295,"inventory":1930},
    {"name":"Clothing - 50","revenue":50,"demand":290,"inventory":1890},
    {"name":"Clothing - 500","revenue":500,"demand":244,"inventory":1570},
    {"name":"Electronics - 25","revenue":25,"demand":273,"inventory":1810},
    {"name":"Electronics - 30","revenue":30,"demand":220,"inventory":1410},
    {"name":"Electronics - 300","revenue":300,"demand":286,"inventory":1830},
    {"name":"Electronics - 50","revenue":50,"demand":268,"inventory":1750},
    {"name":"Electronics - 500","revenue":500,"demand":262,"inventory":1690},
    {"name":"Home Goods - 25","revenue":25,"demand":255,"inventory":1660},
    {"name":"Home Goods - 30","revenue":30,"demand":218,"inventory":1417},
    {"name":"Home Goods - 50","revenue":50,"demand":278,"inventory":1807},
    {"name":"Home Goods - 300","revenue":300,"demand":195,"inventory":1268},
    {"name":"Home Goods - 500","revenue":500,"demand":248,"inventory":1612},
    {"name":"Sports - 25","revenue":25,"demand":269,"inventory":1749},
    {"name":"Sports - 30","revenue":30,"demand":227,"inventory":1476},
    {"name":"Sports - 50","revenue":50,"demand":285,"inventory":1853},
    {"name":"Sports - 300","revenue":300,"demand":208,"inventory":1352},
    {"name":"Sports - 500","revenue":500,"demand":259,"inventory":1684},
    {"name":"Furniture - 25","revenue":25,"demand":242,"inventory":1573},
    {"name":"Furniture - 30","revenue":30,"demand":213,"inventory":1385},
    {"name":"Furniture - 50","revenue":50,"demand":272,"inventory":1768},
    {"name":"Furniture - 300","revenue":300,"demand":224,"inventory":1456},
    {"name":"Furniture - 500","revenue":500,"demand":251,"inventory":1632},
    {"name":"Toys - 25","revenue":25,"demand":257,"inventory":1671},
    {"name":"Toys - 30","revenue":30,"demand":235,"inventory":1528},
    {"name":"Toys - 50","revenue":50,"demand":291,"inventory":1892},
    {"name":"Toys - 300","revenue":300,"demand":199,"inventory":1294},
    {"name":"Toys - 500","revenue":500,"demand":264,"inventory":1716}
  ]
}