Let $I$ be the set of displays (indexed by ShelfID from capacity.csv), and $J$ be the set of products (indexed by ProductName from products.csv). Let $x_{ij}$ be the number of units of product $j$ placed on display $i$.

**Parameters:**

- Displays ($i$):  
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10

- Products ($j$):  
  Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader

- Display capacities (from capacity.csv, by ShelfID):  
  1: 5  
  2: 7  
  3: 6  
  4: 8  
  5: 5.5  
  6: 9  
  7: 6.5  
  8: 7.5  
  9: 8.2  
  10: 5.7

- Product values and weights (from products.csv):

| Product                | Value | Weight |
|------------------------|-------|--------|
| Smartphone             | 200   | 1      |
| Laptop                 | 1500  | 5      |
| Headphones             | 100   | 0.5    |
| Camera                 | 800   | 2      |
| Smartwatch             | 250   | 0.3    |
| Tablet                 | 600   | 1.5    |
| Bluetooth Speaker      | 150   | 1      |
| Keyboard               | 80    | 0.8    |
| Mouse                  | 50    | 0.2    |
| Monitor                | 300   | 3      |
| Printer                | 400   | 4      |
| External Hard Drive    | 120   | 0.5    |
| Router                 | 60    | 0.3    |
| Power Bank             | 40    | 0.4    |
| Memory Card            | 30    | 0.05   |
| USB Flash Drive        | 25    | 0.02   |
| Smart Home Hub         | 100   | 0.6    |
| Gaming Console         | 500   | 4      |
| Fitness Tracker        | 90    | 0.2    |
| E-Reader               | 180   | 0.5    |

**Decision Variables:**

$x_{ij} \in \mathbb{Z}_{\geq 0}$, for all displays $i$ and products $j$.

---

**Objective:**

$$
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in J} v_j \cdot x_{ij}
$$

where $v_j$ is the value of product $j$.

---

**Constraints:**

1. **Display Capacity Constraints:**  
For each display $i$ (ShelfID), the total weight of products placed does not exceed its capacity:

\[
\sum_{j \in J} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
\]

where $w_j$ is the weight of product $j$, and $c_i$ is the capacity of display $i$.

Explicitly, for each display:

- Display 1: $\sum_{j} w_j x_{1j} \leq 5$
- Display 2: $\sum_{j} w_j x_{2j} \leq 7$
- Display 3: $\sum_{j} w_j x_{3j} \leq 6$
- Display 4: $\sum_{j} w_j x_{4j} \leq 8$
- Display 5: $\sum_{j} w_j x_{5j} \leq 5.5$
- Display 6: $\sum_{j} w_j x_{6j} \leq 9$
- Display 7: $\sum_{j} w_j x_{7j} \leq 6.5$
- Display 8: $\sum_{j} w_j x_{8j} \leq 7.5$
- Display 9: $\sum_{j} w_j x_{9j} \leq 8.2$
- Display 10: $\sum_{j} w_j x_{10j} \leq 5.7$

2. **Minimum Quantity of First Product Constraint:**  
The total quantity of the first product ("Smartphone") placed across all displays must be at least 5:

\[
\sum_{i=1}^{10} x_{i,\text{Smartphone}} \geq 5
\]

3. **Nonnegativity and Integrality:**

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in J
\]

---

**Complete Model:**

$$
\begin{align*}
\max\quad & \sum_{i=1}^{10} \Big[ 200\,x_{i,\text{Smartphone}} + 1500\,x_{i,\text{Laptop}} + 100\,x_{i,\text{Headphones}} + 800\,x_{i,\text{Camera}} + 250\,x_{i,\text{Smartwatch}} \\
&\qquad + 600\,x_{i,\text{Tablet}} + 150\,x_{i,\text{Bluetooth Speaker}} + 80\,x_{i,\text{Keyboard}} + 50\,x_{i,\text{Mouse}} + 300\,x_{i,\text{Monitor}} \\
&\qquad + 400\,x_{i,\text{Printer}} + 120\,x_{i,\text{External Hard Drive}} + 60\,x_{i,\text{Router}} + 40\,x_{i,\text{Power Bank}} \\
&\qquad + 30\,x_{i,\text{Memory Card}} + 25\,x_{i,\text{USB Flash Drive}} + 100\,x_{i,\text{Smart Home Hub}} + 500\,x_{i,\text{Gaming Console}} \\
&\qquad + 90\,x_{i,\text{Fitness Tracker}} + 180\,x_{i,\text{E-Reader}} \Big] \\
\text{s.t.}\quad & \sum_{j \in J} w_j x_{1j} \leq 5 \\
& \sum_{j \in J} w_j x_{2j} \leq 7 \\
& \sum_{j \in J} w_j x_{3j} \leq 6 \\
& \sum_{j \in J} w_j x_{4j} \leq 8 \\
& \sum_{j \in J} w_j x_{5j} \leq 5.5 \\
& \sum_{j \in J} w_j x_{6j} \leq 9 \\
& \sum_{j \in J} w_j x_{7j} \leq 6.5 \\
& \sum_{j \in J} w_j x_{8j} \leq 7.5 \\
& \sum_{j \in J} w_j x_{9j} \leq 8.2 \\
& \sum_{j \in J} w_j x_{10j} \leq 5.7 \\
& \sum_{i=1}^{10} x_{i,\text{Smartphone}} \geq 5 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in J
\end{align*}
$$

Where $w_j$ is the weight of product $j$ as given above, and $J$ is the set of all 20 products listed.