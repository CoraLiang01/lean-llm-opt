Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$.

Let $S$ be the set of shelves, indexed by ShelfID:
$$
S = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}
$$

Let $P$ be the set of products, indexed in the order from products.csv:
$$
P = \{\text{Smartphone}, \text{Laptop}, \text{Headphones}, \text{Camera}, \text{Smartwatch}, \text{Tablet}, \text{Bluetooth Speaker}, \text{Keyboard}, \text{Mouse}, \text{Monitor}, \text{Printer}, \text{External Hard Drive}, \text{Router}, \text{Power Bank}, \text{Memory Card}, \text{USB Flash Drive}, \text{Smart Home Hub}, \text{Gaming Console}, \text{Fitness Tracker}, \text{E-Reader}\}
$$

Let $v_j$ be the value of product $j$:

\[
\begin{align*}
v_{\text{Smartphone}} &= 200 \\
v_{\text{Laptop}} &= 1500 \\
v_{\text{Headphones}} &= 100 \\
v_{\text{Camera}} &= 800 \\
v_{\text{Smartwatch}} &= 250 \\
v_{\text{Tablet}} &= 600 \\
v_{\text{Bluetooth Speaker}} &= 150 \\
v_{\text{Keyboard}} &= 80 \\
v_{\text{Mouse}} &= 50 \\
v_{\text{Monitor}} &= 300 \\
v_{\text{Printer}} &= 400 \\
v_{\text{External Hard Drive}} &= 120 \\
v_{\text{Router}} &= 60 \\
v_{\text{Power Bank}} &= 40 \\
v_{\text{Memory Card}} &= 30 \\
v_{\text{USB Flash Drive}} &= 25 \\
v_{\text{Smart Home Hub}} &= 100 \\
v_{\text{Gaming Console}} &= 500 \\
v_{\text{Fitness Tracker}} &= 90 \\
v_{\text{E-Reader}} &= 180 \\
\end{align*}
\]

Let $w_j$ be the weight of product $j$:

\[
\begin{align*}
w_{\text{Smartphone}} &= 1 \\
w_{\text{Laptop}} &= 5 \\
w_{\text{Headphones}} &= 0.5 \\
w_{\text{Camera}} &= 2 \\
w_{\text{Smartwatch}} &= 0.3 \\
w_{\text{Tablet}} &= 1.5 \\
w_{\text{Bluetooth Speaker}} &= 1 \\
w_{\text{Keyboard}} &= 0.8 \\
w_{\text{Mouse}} &= 0.2 \\
w_{\text{Monitor}} &= 3 \\
w_{\text{Printer}} &= 4 \\
w_{\text{External Hard Drive}} &= 0.5 \\
w_{\text{Router}} &= 0.3 \\
w_{\text{Power Bank}} &= 0.4 \\
w_{\text{Memory Card}} &= 0.05 \\
w_{\text{USB Flash Drive}} &= 0.02 \\
w_{\text{Smart Home Hub}} &= 0.6 \\
w_{\text{Gaming Console}} &= 4 \\
w_{\text{Fitness Tracker}} &= 0.2 \\
w_{\text{E-Reader}} &= 0.5 \\
\end{align*}
\]

Let $c_i$ be the capacity of shelf $i$ (from capacity.csv):

\[
\begin{align*}
c_1 &= 5 \\
c_2 &= 7 \\
c_3 &= 6 \\
c_4 &= 8 \\
c_5 &= 5.5 \\
c_6 &= 9 \\
c_7 &= 6.5 \\
c_8 &= 7.5 \\
c_9 &= 8.2 \\
c_{10} &= 5.7 \\
\end{align*}
\]

The model is:

\[
\textbf{Maximize} \quad Z = \sum_{i \in S} \sum_{j \in P} v_j x_{ij}
\]

Subject to:

1. Shelf capacity constraints (for each shelf $i$):

\[
\sum_{j \in P} w_j x_{ij} \leq c_i \qquad \forall i \in S
\]

2. Minimum allocation of the first product ("Smartphone") across all shelves:

\[
\sum_{i \in S} x_{i,\text{Smartphone}} \geq 5
\]

3. Nonnegativity and integrality:

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in S,\, j \in P
\]

Where:

- $x_{ij}$: Number of units of product $j$ placed on shelf $i$ (decision variables, nonnegative integers)
- $v_j$: Value of product $j$ (see above)
- $w_j$: Weight of product $j$ (see above)
- $c_i$: Capacity of shelf $i$ (see above)

All identifiers and coefficients are as retrieved and in original order.