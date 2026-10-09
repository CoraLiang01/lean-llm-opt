**Sets:**
- $i \in \{\text{1}, \text{2}, \text{3}, \text{4}, \text{5}, \text{6}, \text{7}, \text{8}, \text{9}, \text{10}\}$ (ShelfID from "capacity.csv")
- $j \in \{\text{Smartphone}, \text{Laptop}, \text{Headphones}, \text{Camera}, \text{Smartwatch}, \text{Tablet}, \text{Bluetooth Speaker}, \text{Keyboard}, \text{Mouse}, \text{Monitor}, \text{Printer}, \text{External Hard Drive}, \text{Router}, \text{Power Bank}, \text{Memory Card}, \text{USB Flash Drive}, \text{Smart Home Hub}, \text{Gaming Console}, \text{Fitness Tracker}, \text{E-Reader}\}$ (ProductName from "products.csv")

**Parameters:**
- $v_j$ = Value of product $j$ (from "products.csv")
- $w_j$ = Weight of product $j$ (from "products.csv")
- $C_i$ = Capacity of shelf $i$ (from "capacity.csv")

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of product $j$ placed on shelf $i$

**Objective:**
\[
\max \sum_{i \in \{\text{1},\ldots,\text{10}\}} \sum_{j \in \{\text{all products}\}} v_j \cdot x_{ij}
\]

**Subject to:**

For each shelf $i$:
\[
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{\text{1},\ldots,\text{10}\}
\]

For all shelves $i$ and products $j$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

---

**Data (from CSVs, in original order):**

*Shelves and Capacities (capacity.csv):*
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

*Products, Values, and Weights (products.csv):*
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

**Complete Mathematical Model:**

\[
\begin{align*}
\max \quad & \sum_{i=1}^{10} \Big( 200\,x_{i,\text{Smartphone}} + 1500\,x_{i,\text{Laptop}} + 100\,x_{i,\text{Headphones}} + 800\,x_{i,\text{Camera}} + 250\,x_{i,\text{Smartwatch}} \\
& \qquad + 600\,x_{i,\text{Tablet}} + 150\,x_{i,\text{Bluetooth Speaker}} + 80\,x_{i,\text{Keyboard}} + 50\,x_{i,\text{Mouse}} + 300\,x_{i,\text{Monitor}} \\
& \qquad + 400\,x_{i,\text{Printer}} + 120\,x_{i,\text{External Hard Drive}} + 60\,x_{i,\text{Router}} + 40\,x_{i,\text{Power Bank}} \\
& \qquad + 30\,x_{i,\text{Memory Card}} + 25\,x_{i,\text{USB Flash Drive}} + 100\,x_{i,\text{Smart Home Hub}} + 500\,x_{i,\text{Gaming Console}} \\
& \qquad + 90\,x_{i,\text{Fitness Tracker}} + 180\,x_{i,\text{E-Reader}} \Big) \\
\text{s.t.} \quad & \sum_{j} w_j\,x_{i j} \leq C_i \qquad \forall i = 1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\end{align*}
\]

Where for each shelf $i$:

- $C_1 = 5.0$, $C_2 = 7.0$, $C_3 = 6.0$, $C_4 = 8.0$, $C_5 = 5.5$, $C_6 = 9.0$, $C_7 = 6.5$, $C_8 = 7.5$, $C_9 = 8.2$, $C_{10} = 5.7$

And for each product $j$, $w_j$ is as listed above.

All $x_{ij}$ are nonnegative integers.