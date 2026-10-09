Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$.

**Sets and Indices:**
- $i \in \{\text{1}, \text{2}, \text{3}, \text{4}, \text{5}, \text{6}, \text{7}, \text{8}, \text{9}, \text{10}\}$ (ShelfID from capacity.csv)
- $j \in \{\text{Smartphone}, \text{Laptop}, \text{Headphones}, \text{Camera}, \text{Smartwatch}, \text{Tablet}, \text{Bluetooth Speaker}, \text{Keyboard}, \text{Mouse}, \text{Monitor}, \text{Printer}, \text{External Hard Drive}, \text{Router}, \text{Power Bank}, \text{Memory Card}, \text{USB Flash Drive}, \text{Smart Home Hub}, \text{Gaming Console}, \text{Fitness Tracker}, \text{E-Reader}\}$ (ProductName from products.csv)

**Parameters:**

From capacity.csv (in source order):

| ShelfID | Capacity |
|---------|----------|
| 1       | 5        |
| 2       | 7        |
| 3       | 6        |
| 4       | 8        |
| 5       | 5.5      |
| 6       | 9        |
| 7       | 6.5      |
| 8       | 7.5      |
| 9       | 8.2      |
| 10      | 5.7      |

From products.csv (in source order):

| ProductName           | Value | Weight |
|-----------------------|-------|--------|
| Smartphone            | 200   | 1      |
| Laptop                | 1500  | 5      |
| Headphones            | 100   | 0.5    |
| Camera                | 800   | 2      |
| Smartwatch            | 250   | 0.3    |
| Tablet                | 600   | 1.5    |
| Bluetooth Speaker     | 150   | 1      |
| Keyboard              | 80    | 0.8    |
| Mouse                 | 50    | 0.2    |
| Monitor               | 300   | 3      |
| Printer               | 400   | 4      |
| External Hard Drive   | 120   | 0.5    |
| Router                | 60    | 0.3    |
| Power Bank            | 40    | 0.4    |
| Memory Card           | 30    | 0.05   |
| USB Flash Drive       | 25    | 0.02   |
| Smart Home Hub        | 100   | 0.6    |
| Gaming Console        | 500   | 4      |
| Fitness Tracker       | 90    | 0.2    |
| E-Reader              | 180   | 0.5    |

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$, for all shelves $i$ and products $j$

---

### Mathematical Model

**Objective:**
\[
\max \sum_{i \in \{\text{1},\ldots,\text{10}\}} \Bigg(
    200\, x_{i,\text{Smartphone}}
  + 1500\, x_{i,\text{Laptop}}
  + 100\, x_{i,\text{Headphones}}
  + 800\, x_{i,\text{Camera}}
  + 250\, x_{i,\text{Smartwatch}}
  + 600\, x_{i,\text{Tablet}}
  + 150\, x_{i,\text{Bluetooth Speaker}}
  + 80\, x_{i,\text{Keyboard}}
  + 50\, x_{i,\text{Mouse}}
  + 300\, x_{i,\text{Monitor}}
  + 400\, x_{i,\text{Printer}}
  + 120\, x_{i,\text{External Hard Drive}}
  + 60\, x_{i,\text{Router}}
  + 40\, x_{i,\text{Power Bank}}
  + 30\, x_{i,\text{Memory Card}}
  + 25\, x_{i,\text{USB Flash Drive}}
  + 100\, x_{i,\text{Smart Home Hub}}
  + 500\, x_{i,\text{Gaming Console}}
  + 90\, x_{i,\text{Fitness Tracker}}
  + 180\, x_{i,\text{E-Reader}}
\Bigg)
\]

**Subject to:**

For each shelf $i$ (using source order):

- Shelf 1:
  \[
  1\, x_{1,\text{Smartphone}}
  + 5\, x_{1,\text{Laptop}}
  + 0.5\, x_{1,\text{Headphones}}
  + 2\, x_{1,\text{Camera}}
  + 0.3\, x_{1,\text{Smartwatch}}
  + 1.5\, x_{1,\text{Tablet}}
  + 1\, x_{1,\text{Bluetooth Speaker}}
  + 0.8\, x_{1,\text{Keyboard}}
  + 0.2\, x_{1,\text{Mouse}}
  + 3\, x_{1,\text{Monitor}}
  + 4\, x_{1,\text{Printer}}
  + 0.5\, x_{1,\text{External Hard Drive}}
  + 0.3\, x_{1,\text{Router}}
  + 0.4\, x_{1,\text{Power Bank}}
  + 0.05\, x_{1,\text{Memory Card}}
  + 0.02\, x_{1,\text{USB Flash Drive}}
  + 0.6\, x_{1,\text{Smart Home Hub}}
  + 4\, x_{1,\text{Gaming Console}}
  + 0.2\, x_{1,\text{Fitness Tracker}}
  + 0.5\, x_{1,\text{E-Reader}}
  \leq 5
  \]

- Shelf 2:
  \[
  1\, x_{2,\text{Smartphone}}
  + 5\, x_{2,\text{Laptop}}
  + 0.5\, x_{2,\text{Headphones}}
  + 2\, x_{2,\text{Camera}}
  + 0.3\, x_{2,\text{Smartwatch}}
  + 1.5\, x_{2,\text{Tablet}}
  + 1\, x_{2,\text{Bluetooth Speaker}}
  + 0.8\, x_{2,\text{Keyboard}}
  + 0.2\, x_{2,\text{Mouse}}
  + 3\, x_{2,\text{Monitor}}
  + 4\, x_{2,\text{Printer}}
  + 0.5\, x_{2,\text{External Hard Drive}}
  + 0.3\, x_{2,\text{Router}}
  + 0.4\, x_{2,\text{Power Bank}}
  + 0.05\, x_{2,\text{Memory Card}}
  + 0.02\, x_{2,\text{USB Flash Drive}}
  + 0.6\, x_{2,\text{Smart Home Hub}}
  + 4\, x_{2,\text{Gaming Console}}
  + 0.2\, x_{2,\text{Fitness Tracker}}
  + 0.5\, x_{2,\text{E-Reader}}
  \leq 7
  \]

- Shelf 3:
  \[
  1\, x_{3,\text{Smartphone}}
  + 5\, x_{3,\text{Laptop}}
  + 0.5\, x_{3,\text{Headphones}}
  + 2\, x_{3,\text{Camera}}
  + 0.3\, x_{3,\text{Smartwatch}}
  + 1.5\, x_{3,\text{Tablet}}
  + 1\, x_{3,\text{Bluetooth Speaker}}
  + 0.8\, x_{3,\text{Keyboard}}
  + 0.2\, x_{3,\text{Mouse}}
  + 3\, x_{3,\text{Monitor}}
  + 4\, x_{3,\text{Printer}}
  + 0.5\, x_{3,\text{External Hard Drive}}
  + 0.3\, x_{3,\text{Router}}
  + 0.4\, x_{3,\text{Power Bank}}
  + 0.05\, x_{3,\text{Memory Card}}
  + 0.02\, x_{3,\text{USB Flash Drive}}
  + 0.6\, x_{3,\text{Smart Home Hub}}
  + 4\, x_{3,\text{Gaming Console}}
  + 0.2\, x_{3,\text{Fitness Tracker}}
  + 0.5\, x_{3,\text{E-Reader}}
  \leq 6
  \]

- Shelf 4:
  \[
  1\, x_{4,\text{Smartphone}}
  + 5\, x_{4,\text{Laptop}}
  + 0.5\, x_{4,\text{Headphones}}
  + 2\, x_{4,\text{Camera}}
  + 0.3\, x_{4,\text{Smartwatch}}
  + 1.5\, x_{4,\text{Tablet}}
  + 1\, x_{4,\text{Bluetooth Speaker}}
  + 0.8\, x_{4,\text{Keyboard}}
  + 0.2\, x_{4,\text{Mouse}}
  + 3\, x_{4,\text{Monitor}}
  + 4\, x_{4,\text{Printer}}
  + 0.5\, x_{4,\text{External Hard Drive}}
  + 0.3\, x_{4,\text{Router}}
  + 0.4\, x_{4,\text{Power Bank}}
  + 0.05\, x_{4,\text{Memory Card}}
  + 0.02\, x_{4,\text{USB Flash Drive}}
  + 0.6\, x_{4,\text{Smart Home Hub}}
  + 4\, x_{4,\text{Gaming Console}}
  + 0.2\, x_{4,\text{Fitness Tracker}}
  + 0.5\, x_{4,\text{E-Reader}}
  \leq 8
  \]

- Shelf 5:
  \[
  1\, x_{5,\text{Smartphone}}
  + 5\, x_{5,\text{Laptop}}
  + 0.5\, x_{5,\text{Headphones}}
  + 2\, x_{5,\text{Camera}}
  + 0.3\, x_{5,\text{Smartwatch}}
  + 1.5\, x_{5,\text{Tablet}}
  + 1\, x_{5,\text{Bluetooth Speaker}}
  + 0.8\, x_{5,\text{Keyboard}}
  + 0.2\, x_{5,\text{Mouse}}
  + 3\, x_{5,\text{Monitor}}
  + 4\, x_{5,\text{Printer}}
  + 0.5\, x_{5,\text{External Hard Drive}}
  + 0.3\, x_{5,\text{Router}}
  + 0.4\, x_{5,\text{Power Bank}}
  + 0.05\, x_{5,\text{Memory Card}}
  + 0.02\, x_{5,\text{USB Flash Drive}}
  + 0.6\, x_{5,\text{Smart Home Hub}}
  + 4\, x_{5,\text{Gaming Console}}
  + 0.2\, x_{5,\text{Fitness Tracker}}
  + 0.5\, x_{5,\text{E-Reader}}
  \leq 5.5
  \]

- Shelf 6:
  \[
  1\, x_{6,\text{Smartphone}}
  + 5\, x_{6,\text{Laptop}}
  + 0.5\, x_{6,\text{Headphones}}
  + 2\, x_{6,\text{Camera}}
  + 0.3\, x_{6,\text{Smartwatch}}
  + 1.5\, x_{6,\text{Tablet}}
  + 1\, x_{6,\text{Bluetooth Speaker}}
  + 0.8\, x_{6,\text{Keyboard}}
  + 0.2\, x_{6,\text{Mouse}}
  + 3\, x_{6,\text{Monitor}}
  + 4\, x_{6,\text{Printer}}
  + 0.5\, x_{6,\text{External Hard Drive}}
  + 0.3\, x_{6,\text{Router}}
  + 0.4\, x_{6,\text{Power Bank}}
  + 0.05\, x_{6,\text{Memory Card}}
  + 0.02\, x_{6,\text{USB Flash Drive}}
  + 0.6\, x_{6,\text{Smart Home Hub}}
  + 4\, x_{6,\text{Gaming Console}}
  + 0.2\, x_{6,\text{Fitness Tracker}}
  + 0.5\, x_{6,\text{E-Reader}}
  \leq 9
  \]

- Shelf 7:
  \[
  1\, x_{7,\text{Smartphone}}
  + 5\, x_{7,\text{Laptop}}
  + 0.5\, x_{7,\text{Headphones}}
  + 2\, x_{7,\text{Camera}}
  + 0.3\, x_{7,\text{Smartwatch}}
  + 1.5\, x_{7,\text{Tablet}}
  + 1\, x_{7,\text{Bluetooth Speaker}}
  + 0.8\, x_{7,\text{Keyboard}}
  + 0.2\, x_{7,\text{Mouse}}
  + 3\, x_{7,\text{Monitor}}
  + 4\, x_{7,\text{Printer}}
  + 0.5\, x_{7,\text{External Hard Drive}}
  + 0.3\, x_{7,\text{Router}}
  + 0.4\, x_{7,\text{Power Bank}}
  + 0.05\, x_{7,\text{Memory Card}}
  + 0.02\, x_{7,\text{USB Flash Drive}}
  + 0.6\, x_{7,\text{Smart Home Hub}}
  + 4\, x_{7,\text{Gaming Console}}
  + 0.2\, x_{7,\text{Fitness Tracker}}
  + 0.5\, x_{7,\text{E-Reader}}
  \leq 6.5
  \]

- Shelf 8:
  \[
  1\, x_{8,\text{Smartphone}}
  + 5\, x_{8,\text{Laptop}}
  + 0.5\, x_{8,\text{Headphones}}
  + 2\, x_{8,\text{Camera}}
  + 0.3\, x_{8,\text{Smartwatch}}
  + 1.5\, x_{8,\text{Tablet}}
  + 1\, x_{8,\text{Bluetooth Speaker}}
  + 0.8\, x_{8,\text{Keyboard}}
  + 0.2\, x_{8,\text{Mouse}}
  + 3\, x_{8,\text{Monitor}}
  + 4\, x_{8,\text{Printer}}
  + 0.5\, x_{8,\text{External Hard Drive}}
  + 0.3\, x_{8,\text{Router}}
  + 0.4\, x_{8,\text{Power Bank}}
  + 0.05\, x_{8,\text{Memory Card}}
  + 0.02\, x_{8,\text{USB Flash Drive}}
  + 0.6\, x_{8,\text{Smart Home Hub}}
  + 4\, x_{8,\text{Gaming Console}}
  + 0.2\, x_{8,\text{Fitness Tracker}}
  + 0.5\, x_{8,\text{E-Reader}}
  \leq 7.5
  \]

- Shelf 9:
  \[
  1\, x_{9,\text{Smartphone}}
  + 5\, x_{9,\text{Laptop}}
  + 0.5\, x_{9,\text{Headphones}}
  + 2\, x_{9,\text{Camera}}
  + 0.3\, x_{9,\text{Smartwatch}}
  + 1.5\, x_{9,\text{Tablet}}
  + 1\, x_{9,\text{Bluetooth Speaker}}
  + 0.8\, x_{9,\text{Keyboard}}
  + 0.2\, x_{9,\text{Mouse}}
  + 3\, x_{9,\text{Monitor}}
  + 4\, x_{9,\text{Printer}}
  + 0.5\, x_{9,\text{External Hard Drive}}
  + 0.3\, x_{9,\text{Router}}
  + 0.4\, x_{9,\text{Power Bank}}
  + 0.05\, x_{9,\text{Memory Card}}
  + 0.02\, x_{9,\text{USB Flash Drive}}
  + 0.6\, x_{9,\text{Smart Home Hub}}
  + 4\, x_{9,\text{Gaming Console}}
  + 0.2\, x_{9,\text{Fitness Tracker}}
  + 0.5\, x_{9,\text{E-Reader}}
  \leq 8.2
  \]

- Shelf 10:
  \[
  1\, x_{10,\text{Smartphone}}
  + 5\, x_{10,\text{Laptop}}
  + 0.5\, x_{10,\text{Headphones}}
  + 2\, x_{10,\text{Camera}}
  + 0.3\, x_{10,\text{Smartwatch}}
  + 1.5\, x_{10,\text{Tablet}}
  + 1\, x_{10,\text{Bluetooth Speaker}}
  + 0.8\, x_{10,\text{Keyboard}}
  + 0.2\, x_{10,\text{Mouse}}
  + 3\, x_{10,\text{Monitor}}
  + 4\, x_{10,\text{Printer}}
  + 0.5\, x_{10,\text{External Hard Drive}}
  + 0.3\, x_{10,\text{Router}}
  + 0.4\, x_{10,\text{Power Bank}}
  + 0.05\, x_{10,\text{Memory Card}}
  + 0.02\, x_{10,\text{USB Flash Drive}}
  + 0.6\, x_{10,\text{Smart Home Hub}}
  + 4\, x_{10,\text{Gaming Console}}
  + 0.2\, x_{10,\text{Fitness Tracker}}
  + 0.5\, x_{10,\text{E-Reader}}
  \leq 5.7
  \]

**Minimum total quantity of the first product (Smartphone) across all shelves:**
\[
\sum_{i=1}^{10} x_{i,\text{Smartphone}} \geq 5
\]

**Nonnegativity and integrality:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{\text{1},\ldots,\text{10}\},\ j \in \{\text{Smartphone}, \ldots, \text{E-Reader}\}
\]