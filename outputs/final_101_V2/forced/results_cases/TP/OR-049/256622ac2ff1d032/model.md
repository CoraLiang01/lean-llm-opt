##### Sets

Let $I = \{1,2,3,4,5,6,7,8,9,10\}$ be the set of shelves (ShelfID).
Let $J =$ {Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader} be the set of products (ProductName).

##### Parameters

For each shelf $i \in I$:
- $C_i$ = capacity of shelf $i$.

For each product $j \in J$:
- $v_j$ = value of product $j$.
- $w_j$ = weight of product $j$.

##### Decision Variables

For each shelf $i \in I$ and product $j \in J$:
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed on shelf $i$.

##### Objective

Maximize total value across all shelves:
$$
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
$$

##### Constraints

For each shelf $i \in I$:
$$
\sum_{j \in J} w_j x_{ij} \leq C_i
$$

For all $i \in I$, $j \in J$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

##### Data

Shelf capacities (in source order):

\[
\begin{array}{ll}
C_1 = 5.0 & C_6 = 9.0 \\
C_2 = 7.0 & C_7 = 6.5 \\
C_3 = 6.0 & C_8 = 7.5 \\
C_4 = 8.0 & C_9 = 8.2 \\
C_5 = 5.5 & C_{10} = 5.7 \\
\end{array}
\]

Product values and weights (in source order):

\[
\begin{array}{lll}
\text{Product} & v_j & w_j \\
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

##### Complete Model

Maximize
$$
\sum_{i=1}^{10} \Big(
200\,x_{i,\text{Smartphone}} + 1500\,x_{i,\text{Laptop}} + 100\,x_{i,\text{Headphones}} + 800\,x_{i,\text{Camera}} + 250\,x_{i,\text{Smartwatch}} + 600\,x_{i,\text{Tablet}} + 150\,x_{i,\text{Bluetooth Speaker}} + 80\,x_{i,\text{Keyboard}} + 50\,x_{i,\text{Mouse}} + 300\,x_{i,\text{Monitor}} + 400\,x_{i,\text{Printer}} + 120\,x_{i,\text{External Hard Drive}} + 60\,x_{i,\text{Router}} + 40\,x_{i,\text{Power Bank}} + 30\,x_{i,\text{Memory Card}} + 25\,x_{i,\text{USB Flash Drive}} + 100\,x_{i,\text{Smart Home Hub}} + 500\,x_{i,\text{Gaming Console}} + 90\,x_{i,\text{Fitness Tracker}} + 180\,x_{i,\text{E-Reader}}
\Big)
$$

Subject to, for each $i=1,\ldots,10$:
$$
1.0\,x_{i,\text{Smartphone}} + 5.0\,x_{i,\text{Laptop}} + 0.5\,x_{i,\text{Headphones}} + 2.0\,x_{i,\text{Camera}} + 0.3\,x_{i,\text{Smartwatch}} + 1.5\,x_{i,\text{Tablet}} + 1.0\,x_{i,\text{Bluetooth Speaker}} + 0.8\,x_{i,\text{Keyboard}} + 0.2\,x_{i,\text{Mouse}} + 3.0\,x_{i,\text{Monitor}} + 4.0\,x_{i,\text{Printer}} + 0.5\,x_{i,\text{External Hard Drive}} + 0.3\,x_{i,\text{Router}} + 0.4\,x_{i,\text{Power Bank}} + 0.05\,x_{i,\text{Memory Card}} + 0.02\,x_{i,\text{USB Flash Drive}} + 0.6\,x_{i,\text{Smart Home Hub}} + 4.0\,x_{i,\text{Gaming Console}} + 0.2\,x_{i,\text{Fitness Tracker}} + 0.5\,x_{i,\text{E-Reader}} \leq C_i
$$

For all $i=1,\ldots,10$, $j \in J$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$