**Sets and Indices:**  
- $i \in \{\text{1}, \text{2}, \text{3}, \text{4}, \text{5}, \text{6}, \text{7}, \text{8}, \text{9}, \text{10}\}$ (ShelfID)  
- $j \in \{\text{Smartphone}, \text{Laptop}, \text{Headphones}, \text{Camera}, \text{Smartwatch}, \text{Tablet}, \text{Bluetooth Speaker}, \text{Keyboard}, \text{Mouse}, \text{Monitor}, \text{Printer}, \text{External Hard Drive}, \text{Router}, \text{Power Bank}, \text{Memory Card}, \text{USB Flash Drive}, \text{Smart Home Hub}, \text{Gaming Console}, \text{Fitness Tracker}, \text{E-Reader}\}$ (ProductName)

**Parameters:**  
- $v_j$ = Value of product $j$ (see table below)  
- $w_j$ = Weight of product $j$ (see table below)  
- $C_i$ = Capacity of display $i$ (see table below)

**Decision Variables:**  
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of product $j$ placed on display $i$

---

**Objective:**  
$$
\max \sum_{i \in \{\text{1},\ldots,\text{10}\}} \sum_{j \in \{\text{Smartphone},\ldots,\text{E-Reader}\}} v_j \cdot x_{ij}
$$

**Subject to:**

1. **Display Capacity Constraints:**  
For each display $i$:
$$
\sum_{j \in \{\text{Smartphone},\ldots,\text{E-Reader}\}} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{\text{1},\ldots,\text{10}\}
$$

2. **Minimum Quantity for First Product ("Smartphone"):**
$$
\sum_{i \in \{\text{1},\ldots,\text{10}\}} x_{i,\text{Smartphone}} \geq 5
$$

3. **Nonnegativity and Integrality:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

---

**Parameter Table (from CSV):**

| ProductName             | Value | Weight |
|-------------------------|-------|--------|
| Smartphone              | 200   | 1.0    |
| Laptop                  | 1500  | 5.0    |
| Headphones              | 100   | 0.5    |
| Camera                  | 800   | 2.0    |
| Smartwatch              | 250   | 0.3    |
| Tablet                  | 600   | 1.5    |
| Bluetooth Speaker       | 150   | 1.0    |
| Keyboard                | 80    | 0.8    |
| Mouse                   | 50    | 0.2    |
| Monitor                 | 300   | 3.0    |
| Printer                 | 400   | 4.0    |
| External Hard Drive     | 120   | 0.5    |
| Router                  | 60    | 0.3    |
| Power Bank              | 40    | 0.4    |
| Memory Card             | 30    | 0.05   |
| USB Flash Drive         | 25    | 0.02   |
| Smart Home Hub          | 100   | 0.6    |
| Gaming Console          | 500   | 4.0    |
| Fitness Tracker         | 90    | 0.2    |
| E-Reader                | 180   | 0.5    |

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