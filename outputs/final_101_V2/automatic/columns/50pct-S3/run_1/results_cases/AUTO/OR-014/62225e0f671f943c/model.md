Let $x_{ij}$ be the number of units of product $j$ to be placed on shelf $i$, where $i$ indexes shelves (ShelfID from capacity.csv) and $j$ indexes products (ProductName from products.csv). All $x_{ij}$ are nonnegative integers.

Objective:
$$
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{\text{Smartphone}, \text{Laptop}, \text{Headphones}, \text{Camera}, \text{Smartwatch}, \text{Tablet}, \text{Bluetooth Speaker}, \text{Keyboard}, \text{Mouse}, \text{Monitor}, \text{Printer}, \text{External Hard Drive}, \text{Router}, \text{Power Bank}, \text{Memory Card}, \text{USB Flash Drive}, \text{Smart Home Hub}, \text{Gaming Console}, \text{Fitness Tracker}, \text{E-Reader}\}} v_j \cdot x_{ij}
$$
where $v_j$ is the Value of product $j$ from products.csv.

Subject to, for each shelf $i$ (ShelfID from capacity.csv):

Capacity constraints:
$$
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
$$
where $w_j$ is the Weight of product $j$ from products.csv, and $C_i$ is the Capacity of shelf $i$ from capacity.csv.

Integrality and nonnegativity:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

Numerical Data:

Shelves (from capacity.csv, in order):

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

Products (from products.csv, in order):

| ProductName           | Value | Weight |
|-----------------------|-------|--------|
| Smartphone            | 200   | 1.0    |
| Laptop                | 1500  | 5.0    |
| Headphones            | 100   | 0.5    |
| Camera                | 800   | 2.0    |
| Smartwatch            | 250   | 0.3    |
| Tablet                | 600   | 1.5    |
| Bluetooth Speaker     | 150   | 1.0    |
| Keyboard              | 80    | 0.8    |
| Mouse                 | 50    | 0.2    |
| Monitor               | 300   | 3.0    |
| Printer               | 400   | 4.0    |
| External Hard Drive   | 120   | 0.5    |
| Router                | 60    | 0.3    |
| Power Bank            | 40    | 0.4    |
| Memory Card           | 30    | 0.05   |
| USB Flash Drive       | 25    | 0.02   |
| Smart Home Hub        | 100   | 0.6    |
| Gaming Console        | 500   | 4.0    |
| Fitness Tracker       | 90    | 0.2    |
| E-Reader              | 180   | 0.5    |

Full Model:

Maximize
$$
\sum_{i=1}^{10} \Big(
200\,x_{i,\text{Smartphone}} + 1500\,x_{i,\text{Laptop}} + 100\,x_{i,\text{Headphones}} + 800\,x_{i,\text{Camera}} + 250\,x_{i,\text{Smartwatch}} + 600\,x_{i,\text{Tablet}} + 150\,x_{i,\text{Bluetooth Speaker}} + 80\,x_{i,\text{Keyboard}} + 50\,x_{i,\text{Mouse}} + 300\,x_{i,\text{Monitor}} + 400\,x_{i,\text{Printer}} + 120\,x_{i,\text{External Hard Drive}} + 60\,x_{i,\text{Router}} + 40\,x_{i,\text{Power Bank}} + 30\,x_{i,\text{Memory Card}} + 25\,x_{i,\text{USB Flash Drive}} + 100\,x_{i,\text{Smart Home Hub}} + 500\,x_{i,\text{Gaming Console}} + 90\,x_{i,\text{Fitness Tracker}} + 180\,x_{i,\text{E-Reader}}
\Big)
$$

Subject to, for each $i=1,\ldots,10$:
$$
1.0\,x_{i,\text{Smartphone}} + 5.0\,x_{i,\text{Laptop}} + 0.5\,x_{i,\text{Headphones}} + 2.0\,x_{i,\text{Camera}} + 0.3\,x_{i,\text{Smartwatch}} + 1.5\,x_{i,\text{Tablet}} + 1.0\,x_{i,\text{Bluetooth Speaker}} + 0.8\,x_{i,\text{Keyboard}} + 0.2\,x_{i,\text{Mouse}} + 3.0\,x_{i,\text{Monitor}} + 4.0\,x_{i,\text{Printer}} + 0.5\,x_{i,\text{External Hard Drive}} + 0.3\,x_{i,\text{Router}} + 0.4\,x_{i,\text{Power Bank}} + 0.05\,x_{i,\text{Memory Card}} + 0.02\,x_{i,\text{USB Flash Drive}} + 0.6\,x_{i,\text{Smart Home Hub}} + 4.0\,x_{i,\text{Gaming Console}} + 0.2\,x_{i,\text{Fitness Tracker}} + 0.5\,x_{i,\text{E-Reader}} \leq \text{Capacity}_i
$$
where $\text{Capacity}_i$ is as listed above for each shelf.

And
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$