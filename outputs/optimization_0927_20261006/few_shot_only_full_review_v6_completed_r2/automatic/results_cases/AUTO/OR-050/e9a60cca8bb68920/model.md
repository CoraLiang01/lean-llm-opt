**Sets and Indices:**  
- $i$ indexes displays (ShelfID): $i \in \{1,2,3,4,5,6,7,8,9,10\}$  
- $j$ indexes products (ProductName):  
  $j \in$ {Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader}

**Parameters:**  
- $C_i$ = Capacity of display $i$ (from capacity.csv)  
- $v_j$ = Value of product $j$ (from products.csv)  
- $w_j$ = Weight of product $j$ (from products.csv)  

**Decision Variables:**  
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of product $j$ placed on display $i$

**Objective:**  
Maximize total value:
$$
\max \sum_{i \in \{1,2,3,4,5,6,7,8,9,10\}} \sum_{j \in \text{Products}} v_j x_{ij}
$$

**Subject to:**

1. **Display Capacity Constraints:**  
For each display $i$:
$$
\sum_{j \in \text{Products}} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,2,3,4,5,6,7,8,9,10\}
$$

2. **Minimum Smartphone Allocation Constraint:**  
$$
\sum_{i \in \{1,2,3,4,5,6,7,8,9,10\}} x_{i,\text{Smartphone}} \geq 5
$$

3. **Nonnegativity and Integrality:**  
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

---

**Parameter Data (from CSVs):**

*Displays (capacity.csv):*

| ShelfID | Capacity |
|---------|----------|
| 1       | 5.0      |
| 2       | 7.0      |
| 3       | 6.0      |
| 4       | 8.0      |
| 5       | 5.5      |
| 6       | 9.0      |
| 7       | 6.5      |
| 8       | 7.5      |
| 9       | 8.2      |
| 10      | 5.7      |

*Products (products.csv):*

| ProductName            | Value | Weight |
|------------------------|-------|--------|
| Smartphone             | 200   | 1.0    |
| Laptop                 | 1500  | 5.0    |
| Headphones             | 100   | 0.5    |
| Camera                 | 800   | 2.0    |
| Smartwatch             | 250   | 0.3    |
| Tablet                 | 600   | 1.5    |
| Bluetooth Speaker      | 150   | 1.0    |
| Keyboard               | 80    | 0.8    |
| Mouse                  | 50    | 0.2    |
| Monitor                | 300   | 3.0    |
| Printer                | 400   | 4.0    |
| External Hard Drive    | 120   | 0.5    |
| Router                 | 60    | 0.3    |
| Power Bank             | 40    | 0.4    |
| Memory Card            | 30    | 0.05   |
| USB Flash Drive        | 25    | 0.02   |
| Smart Home Hub         | 100   | 0.6    |
| Gaming Console         | 500   | 4.0    |
| Fitness Tracker        | 90    | 0.2    |
| E-Reader               | 180   | 0.5    |

---

**Full Model:**

Let $x_{ij}$ be the number of units of product $j$ placed on display $i$.

$$
\max \sum_{i=1}^{10} \Bigg[
200\,x_{i,\text{Smartphone}} + 1500\,x_{i,\text{Laptop}} + 100\,x_{i,\text{Headphones}} + 800\,x_{i,\text{Camera}} + 250\,x_{i,\text{Smartwatch}} + 600\,x_{i,\text{Tablet}} + 150\,x_{i,\text{Bluetooth Speaker}} + 80\,x_{i,\text{Keyboard}} + 50\,x_{i,\text{Mouse}} + 300\,x_{i,\text{Monitor}} + 400\,x_{i,\text{Printer}} + 120\,x_{i,\text{External Hard Drive}} + 60\,x_{i,\text{Router}} + 40\,x_{i,\text{Power Bank}} + 30\,x_{i,\text{Memory Card}} + 25\,x_{i,\text{USB Flash Drive}} + 100\,x_{i,\text{Smart Home Hub}} + 500\,x_{i,\text{Gaming Console}} + 90\,x_{i,\text{Fitness Tracker}} + 180\,x_{i,\text{E-Reader}}
\Bigg]
$$

Subject to, for each $i = 1,\ldots,10$:

$$
1.0\,x_{i,\text{Smartphone}} + 5.0\,x_{i,\text{Laptop}} + 0.5\,x_{i,\text{Headphones}} + 2.0\,x_{i,\text{Camera}} + 0.3\,x_{i,\text{Smartwatch}} + 1.5\,x_{i,\text{Tablet}} + 1.0\,x_{i,\text{Bluetooth Speaker}} + 0.8\,x_{i,\text{Keyboard}} + 0.2\,x_{i,\text{Mouse}} + 3.0\,x_{i,\text{Monitor}} + 4.0\,x_{i,\text{Printer}} + 0.5\,x_{i,\text{External Hard Drive}} + 0.3\,x_{i,\text{Router}} + 0.4\,x_{i,\text{Power Bank}} + 0.05\,x_{i,\text{Memory Card}} + 0.02\,x_{i,\text{USB Flash Drive}} + 0.6\,x_{i,\text{Smart Home Hub}} + 4.0\,x_{i,\text{Gaming Console}} + 0.2\,x_{i,\text{Fitness Tracker}} + 0.5\,x_{i,\text{E-Reader}} \leq C_i
$$

where $C_i$ is the capacity for display $i$ as given above.

And:

$$
\sum_{i=1}^{10} x_{i,\text{Smartphone}} \geq 5
$$

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$