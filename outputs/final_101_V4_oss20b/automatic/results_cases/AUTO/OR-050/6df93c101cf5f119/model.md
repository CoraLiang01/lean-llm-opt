##### Objective Function: 

$\quad \quad \max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}$

##### Constraints

###### 1. Capacity Constraints: 

$\quad \quad \sum_{j=1}^{20} w_j \, x_{ij} \leq C_i , \quad \forall i=1,\dots ,10$

###### 2. Minimum Quantity of First Product: 

$\quad \quad \sum_{i=1}^{10} x_{i1} \geq 5$

###### 3. Non‑negativity and Integrality Constraints: 

$\quad \quad x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i=1,\dots ,10,\; \forall j=1,\dots ,20$

---

**Parameters**

- $C_1 = 5.0,\; C_2 = 7.0,\; C_3 = 6.0,\; C_4 = 8.0,\; C_5 = 5.5,\; C_6 = 9.0,\; C_7 = 6.5,\; C_8 = 7.5,\; C_9 = 8.2,\; C_{10} = 5.7$

- $v_1 = 200,\; v_2 = 1500,\; v_3 = 100,\; v_4 = 800,\; v_5 = 250,\; v_6 = 600,\; v_7 = 150,\; v_8 = 80,\; v_9 = 50,\; v_{10} = 300,\; v_{11} = 400,\; v_{12} = 120,\; v_{13} = 60,\; v_{14} = 40,\; v_{15} = 30,\; v_{16} = 25,\; v_{17} = 100,\; v_{18} = 500,\; v_{19} = 90,\; v_{20} = 180$

- $w_1 = 1.0,\; w_2 = 5.0,\; w_3 = 0.5,\; w_4 = 2.0,\; w_5 = 0.3,\; w_6 = 1.5,\; w_7 = 1.0,\; w_8 = 0.8,\; w_9 = 0.2,\; w_{10} = 3.0,\; w_{11} = 4.0,\; w_{12} = 0.5,\; w_{13} = 0.3,\; w_{14} = 0.4,\; w_{15} = 0.05,\; w_{16} = 0.02,\; w_{17} = 0.6,\; w_{18} = 4.0,\; w_{19} = 0.2,\; w_{20} = 0.5$