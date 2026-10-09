**Sets:**
- $i \in \{1,2,3,4,5,6,7,8,9,10\}$ (ShelfID from capacity.csv)
- $j \in \{$Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader$\}$ (ProductName from products.csv)

**Parameters:**
- $C_i$: Capacity of display $i$ (from capacity.csv)
- $v_j$: Value of product $j$ (from products.csv)
- $w_j$: Weight of product $j$ (from products.csv)

**Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of product $j$ placed on display $i$

**Data:**

From capacity.csv:
| ShelfID ($i$) | $C_i$ |
|---|---|
| 1 | 5.0 |
| 2 | 7.0 |
| 3 | 6.0 |
| 4 | 8.0 |
| 5 | 5.5 |
| 6 | 9.0 |
| 7 | 6.5 |
| 8 | 7.5 |
| 9 | 8.2 |
| 10 | 5.7 |

From products.csv:
| $j$ | $v_j$ | $w_j$ |
|---|---|---|
| Smartphone | 200 | 1.0 |
| Laptop | 1500 | 5.0 |
| Headphones | 100 | 0.5 |
| Camera | 800 | 2.0 |
| Smartwatch | 250 | 0.3 |
| Tablet | 600 | 1.5 |
| Bluetooth Speaker | 150 | 1.0 |
| Keyboard | 80 | 0.8 |
| Mouse | 50 | 0.2 |
| Monitor | 300 | 3.0 |
| Printer | 400 | 4.0 |
| External Hard Drive | 120 | 0.5 |
| Router | 60 | 0.3 |
| Power Bank | 40 | 0.4 |
| Memory Card | 30 | 0.05 |
| USB Flash Drive | 25 | 0.02 |
| Smart Home Hub | 100 | 0.6 |
| Gaming Console | 500 | 4.0 |
| Fitness Tracker | 90 | 0.2 |
| E-Reader | 180 | 0.5 |

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
\]

**Subject to:**

1. **Display Capacity Constraints (for each display $i$):**
\[
\sum_{j=1}^{20} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,10\}
\]

2. **Minimum Smartphone Allocation:**
\[
\sum_{i=1}^{10} x_{i, \text{Smartphone}} \geq 5
\]

3. **Nonnegativity and Integrality:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,2,\ldots,10\},\; j \in \{1,2,\ldots,20\}
\]

Where:
- $v_j$ and $w_j$ are as given in the table above for each product $j$.
- $C_i$ is as given in the table above for each display $i$.
- $x_{i, \text{Smartphone}}$ refers to the variable for product "Smartphone" on display $i$.

**Decision variables:**  
$x_{ij}$: number of units of product $j$ placed on display $i$, integer and $\geq 0$.