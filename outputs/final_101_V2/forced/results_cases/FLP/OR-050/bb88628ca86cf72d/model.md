##### Decision Variables

$x_{ij} \geq 0$ and integer: Number of units of product $j$ placed on display $i$, for each display $i \in D$ and product $j \in P$.

##### Parameters

Displays $D = \{1,2,3,4,5,6,7,8,9,10\}$, with capacities:
- $C_1 = 5.0$
- $C_2 = 7.0$
- $C_3 = 6.0$
- $C_4 = 8.0$
- $C_5 = 5.5$
- $C_6 = 9.0$
- $C_7 = 6.5$
- $C_8 = 7.5$
- $C_9 = 8.2$
- $C_{10} = 5.7$

Products $P = \{$
1: Smartphone, 
2: Laptop, 
3: Headphones, 
4: Camera, 
5: Smartwatch, 
6: Tablet, 
7: Bluetooth Speaker, 
8: Keyboard, 
9: Mouse, 
10: Monitor, 
11: Printer, 
12: External Hard Drive, 
13: Router, 
14: Power Bank, 
15: Memory Card, 
16: USB Flash Drive, 
17: Smart Home Hub, 
18: Gaming Console, 
19: Fitness Tracker, 
20: E-Reader
$\}$

Product values $v_j$ and weights $w_j$:
- Smartphone: $v_1 = 200$, $w_1 = 1.0$
- Laptop: $v_2 = 1500$, $w_2 = 5.0$
- Headphones: $v_3 = 100$, $w_3 = 0.5$
- Camera: $v_4 = 800$, $w_4 = 2.0$
- Smartwatch: $v_5 = 250$, $w_5 = 0.3$
- Tablet: $v_6 = 600$, $w_6 = 1.5$
- Bluetooth Speaker: $v_7 = 150$, $w_7 = 1.0$
- Keyboard: $v_8 = 80$, $w_8 = 0.8$
- Mouse: $v_9 = 50$, $w_9 = 0.2$
- Monitor: $v_{10} = 300$, $w_{10} = 3.0$
- Printer: $v_{11} = 400$, $w_{11} = 4.0$
- External Hard Drive: $v_{12} = 120$, $w_{12} = 0.5$
- Router: $v_{13} = 60$, $w_{13} = 0.3$
- Power Bank: $v_{14} = 40$, $w_{14} = 0.4$
- Memory Card: $v_{15} = 30$, $w_{15} = 0.05$
- USB Flash Drive: $v_{16} = 25$, $w_{16} = 0.02$
- Smart Home Hub: $v_{17} = 100$, $w_{17} = 0.6$
- Gaming Console: $v_{18} = 500$, $w_{18} = 4.0$
- Fitness Tracker: $v_{19} = 90$, $w_{19} = 0.2$
- E-Reader: $v_{20} = 180$, $w_{20} = 0.5$

##### Objective Function

$$
\max \sum_{i \in D} \sum_{j \in P} v_j x_{ij}
$$

##### Constraints

1. **Display capacity constraints** (for each display $i$):
   $$
   \sum_{j \in P} w_j x_{ij} \leq C_i, \quad \forall i \in D
   $$

2. **Minimum total quantity of the first product (Smartphone) across all displays**:
   $$
   \sum_{i \in D} x_{i1} \geq 5
   $$

3. **Nonnegativity and integrality**:
   $$
   x_{ij} \geq 0, \quad x_{ij} \in \mathbb{Z}, \quad \forall i \in D,\, j \in P
   $$

##### Summary of Sets and Parameters

- $D = \{1,2,3,4,5,6,7,8,9,10\}$ (displays)
- $P = \{1,2,\ldots,20\}$ (products, as listed above)
- $C_i$ as above for each display $i$
- $v_j$, $w_j$ as above for each product $j$

##### Complete Model

$$
\begin{align*}
\max\quad & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.}\quad & \sum_{j=1}^{20} w_j x_{ij} \leq C_i, \quad \forall i=1,\ldots,10 \\
& \sum_{i=1}^{10} x_{i1} \geq 5 \\
& x_{ij} \geq 0,\ x_{ij} \in \mathbb{Z}, \quad \forall i=1,\ldots,10,\ j=1,\ldots,20
\end{align*}
$$

Where all parameters and sets are as listed above.