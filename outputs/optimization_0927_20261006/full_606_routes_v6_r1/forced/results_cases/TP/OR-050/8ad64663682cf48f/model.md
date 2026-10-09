Let $I = \{1,2,3,4,5,6,7,8,9,10\}$ be the set of displays (ShelfID in source order), and $J = \{$Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader$\}$ be the set of products (ProductName in source order).

Let $x_{ij} \geq 0$ be the number of units of product $j \in J$ placed on display $i \in I$ (continuous or integer as appropriate).

Parameters (from source order):

Display capacities:
\[
\begin{align*}
\text{Capacity}_1 &= 5.0 \\
\text{Capacity}_2 &= 7.0 \\
\text{Capacity}_3 &= 6.0 \\
\text{Capacity}_4 &= 8.0 \\
\text{Capacity}_5 &= 5.5 \\
\text{Capacity}_6 &= 9.0 \\
\text{Capacity}_7 &= 6.5 \\
\text{Capacity}_8 &= 7.5 \\
\text{Capacity}_9 &= 8.2 \\
\text{Capacity}_{10} &= 5.7 \\
\end{align*}
\]

Product values and weights (in source order):

\[
\begin{array}{lll}
\text{Product} & \text{Value} & \text{Weight} \\
\hline
\text{Smartphone} & 200 & 1.0 \\
\text{Laptop} & 1500 & 5.0 \\
\text{Headphones} & 100 & 0.5 \\
\text{Camera} & 800 & 2.0 \\
\text{Smartwatch} & 250 & 0.3 \\
\text{Tablet} & 600 & 1.5 \\
\text{Bluetooth Speaker} & 150 & 1.0 \\
\text{Keyboard} & 80 & 0.8 \\
\text{Mouse} & 50 & 0.2 \\
\text{Monitor} & 300 & 3.0 \\
\text{Printer} & 400 & 4.0 \\
\text{External Hard Drive} & 120 & 0.5 \\
\text{Router} & 60 & 0.3 \\
\text{Power Bank} & 40 & 0.4 \\
\text{Memory Card} & 30 & 0.05 \\
\text{USB Flash Drive} & 25 & 0.02 \\
\text{Smart Home Hub} & 100 & 0.6 \\
\text{Gaming Console} & 500 & 4.0 \\
\text{Fitness Tracker} & 90 & 0.2 \\
\text{E-Reader} & 180 & 0.5 \\
\end{array}
\]

Model:

Maximize total value:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
\]
where $v_j$ is the value of product $j$ as listed above.

Subject to:

1. Capacity constraints for each display:
\[
\sum_{j=1}^{20} w_j x_{ij} \leq \text{Capacity}_i \qquad \forall i=1,\ldots,10
\]
where $w_j$ is the weight of product $j$ as listed above.

2. Minimum total quantity of the first product (Smartphone) across all displays:
\[
\sum_{i=1}^{10} x_{i,1} \geq 5
\]

3. Non-negativity:
\[
x_{ij} \geq 0 \qquad \forall i=1,\ldots,10,\ j=1,\ldots,20
\]

All parameters and indices are in the order as retrieved from the source files.

Retrieved Information:

Displays (ShelfID, Capacity):
1: 5.0  
2: 7.0  
3: 6.0  
4: 8.0  
5: 5.5  
6: 9.0  
7: 6.5  
8: 7.5  
9: 8.2  
10: 5.7  

Products (ProductName, Value, Weight):
1: Smartphone, 200, 1.0  
2: Laptop, 1500, 5.0  
3: Headphones, 100, 0.5  
4: Camera, 800, 2.0  
5: Smartwatch, 250, 0.3  
6: Tablet, 600, 1.5  
7: Bluetooth Speaker, 150, 1.0  
8: Keyboard, 80, 0.8  
9: Mouse, 50, 0.2  
10: Monitor, 300, 3.0  
11: Printer, 400, 4.0  
12: External Hard Drive, 120, 0.5  
13: Router, 60, 0.3  
14: Power Bank, 40, 0.4  
15: Memory Card, 30, 0.05  
16: USB Flash Drive, 25, 0.02  
17: Smart Home Hub, 100, 0.6  
18: Gaming Console, 500, 4.0  
19: Fitness Tracker, 90, 0.2  
20: E-Reader, 180, 0.5