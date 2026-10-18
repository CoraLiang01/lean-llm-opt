##### Decision Variables

$x_{ij} \geq 0$: number of units of product $j \in J$ placed on display $i \in I$ (continuous).

##### Objective Function

$\max \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij}$

##### Constraints

1. Display capacity: $\displaystyle \sum_{j\in J} w_j\,x_{ij}\leq C_i,\quad \forall i\in I$  
2. Minimum first product: $\displaystyle \sum_{i\in I}x_{i1}\geq 5$  
3. Domains: $x_{ij}\geq 0$ for all $i\in I,\;j\in J$.

##### Sets and Parameters

$I=\{1,2,3,4,5,6,7,8,9,10\}$  

$J=\{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20\}$  

$C_i$ (capacity of display $i$):  

$C_1=5.0,\;C_2=7.0,\;C_3=6.0,\;C_4=8.0,\;C_5=5.5,\;C_6=9.0,\;C_7=6.5,\;C_8=7.5,\;C_9=8.2,\;C_{10}=5.7$  

$w_j$ (weight of product $j$):  

$w_1=1.0,\;w_2=5.0,\;w_3=0.5,\;w_4=2.0,\;w_5=0.3,\;w_6=1.5,\;w_7=1.0,\;w_8=0.8,\;w_9=0.2,\;w_{10}=3.0,\;w_{11}=4.0,\;w_{12}=0.5,\;w_{13}=0.3,\;w_{14}=0.4,\;w_{15}=0.05,\;w_{16}=0.02,\;w_{17}=0.6,\;w_{18}=4.0,\;w_{19}=0.2,\;w_{20}=0.5$  

$v_j$ (value of product $j$):  

$v_1=200,\;v_2=1500,\;v_3=100,\;v_4=800,\;v_5=250,\;v_6=600,\;v_7=150,\;v_8=80,\;v_9=50,\;v_{10}=300,\;v_{11}=400,\;v_{12}=120,\;v_{13}=60,\;v_{14}=40,\;v_{15}=30,\;v_{16}=25,\;v_{17}=100,\;v_{18}=500,\;v_{19}=90,\;v_{20}=180$  

##### Retrieved Information

**Displays (I)**: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10  
**Capacities (C)**: 5.0, 7.0, 6.0, 8.0, 5.5, 9.0, 6.5, 7.5, 8.2, 5.7  

**Products (J)**: 1–20  
**Product Names**: Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader  
**Values (v)**: 200, 1500, 100, 800, 250, 600, 150, 80, 50, 300, 400, 120, 60, 40, 30, 25, 100, 500, 90, 180  
**Weights (w)**: 1.0, 5.0, 0.5, 2.0, 0.3, 1.5, 1.0, 0.8, 0.2, 3.0, 4.0, 0.5, 0.3, 0.4, 0.05, 0.02, 0.6, 4.0, 0.2, 0.5