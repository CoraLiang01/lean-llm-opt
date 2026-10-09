**Sets and Indices:**

- Let $i$ index shelves, with ShelfID from capacity.csv: $i \in \{1,2,3,4,5,6,7,8,9,10\}$
- Let $j$ index products, with ProductName from products.csv: $j \in \{\text{Smartphone}, \text{Laptop}, \text{Headphones}, \text{Camera}, \text{Smartwatch}, \text{Tablet}, \text{Bluetooth Speaker}, \text{Keyboard}, \text{Mouse}, \text{Monitor}, \text{Printer}, \text{External Hard Drive}, \text{Router}, \text{Power Bank}, \text{Memory Card}, \text{USB Flash Drive}, \text{Smart Home Hub}, \text{Gaming Console}, \text{Fitness Tracker}, \text{E-Reader}\}$

**Parameters:**

- $v_j$ = Value of product $j$ (from products.csv)
- $w_j$ = Weight of product $j$ (from products.csv)
- $C_i$ = Capacity of shelf $i$ (from capacity.csv)

**Decision Variables:**

- $x_{ij}$ = Number of units of product $j$ placed on shelf $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

---

**Objective:**

$$
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{1,\ldots,20\}} v_j \cdot x_{ij}
$$

Where $v_j$ is as follows (in order of products.csv):

\[
\begin{align*}
v_1 &= 200 \quad &(\text{Smartphone}) \\
v_2 &= 1500 \quad &(\text{Laptop}) \\
v_3 &= 100 \quad &(\text{Headphones}) \\
v_4 &= 800 \quad &(\text{Camera}) \\
v_5 &= 250 \quad &(\text{Smartwatch}) \\
v_6 &= 600 \quad &(\text{Tablet}) \\
v_7 &= 150 \quad &(\text{Bluetooth Speaker}) \\
v_8 &= 80 \quad &(\text{Keyboard}) \\
v_9 &= 50 \quad &(\text{Mouse}) \\
v_{10} &= 300 \quad &(\text{Monitor}) \\
v_{11} &= 400 \quad &(\text{Printer}) \\
v_{12} &= 120 \quad &(\text{External Hard Drive}) \\
v_{13} &= 60 \quad &(\text{Router}) \\
v_{14} &= 40 \quad &(\text{Power Bank}) \\
v_{15} &= 30 \quad &(\text{Memory Card}) \\
v_{16} &= 25 \quad &(\text{USB Flash Drive}) \\
v_{17} &= 100 \quad &(\text{Smart Home Hub}) \\
v_{18} &= 500 \quad &(\text{Gaming Console}) \\
v_{19} &= 90 \quad &(\text{Fitness Tracker}) \\
v_{20} &= 180 \quad &(\text{E-Reader}) \\
\end{align*}
\]

---

**Constraints:**

For each shelf $i$ (with ShelfID and Capacity from capacity.csv):

\[
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,3,4,5,6,7,8,9,10\}
\]

Where $w_j$ is as follows (in order of products.csv):

\[
\begin{align*}
w_1 &= 1.0 \quad &(\text{Smartphone}) \\
w_2 &= 5.0 \quad &(\text{Laptop}) \\
w_3 &= 0.5 \quad &(\text{Headphones}) \\
w_4 &= 2.0 \quad &(\text{Camera}) \\
w_5 &= 0.3 \quad &(\text{Smartwatch}) \\
w_6 &= 1.5 \quad &(\text{Tablet}) \\
w_7 &= 1.0 \quad &(\text{Bluetooth Speaker}) \\
w_8 &= 0.8 \quad &(\text{Keyboard}) \\
w_9 &= 0.2 \quad &(\text{Mouse}) \\
w_{10} &= 3.0 \quad &(\text{Monitor}) \\
w_{11} &= 4.0 \quad &(\text{Printer}) \\
w_{12} &= 0.5 \quad &(\text{External Hard Drive}) \\
w_{13} &= 0.3 \quad &(\text{Router}) \\
w_{14} &= 0.4 \quad &(\text{Power Bank}) \\
w_{15} &= 0.05 \quad &(\text{Memory Card}) \\
w_{16} &= 0.02 \quad &(\text{USB Flash Drive}) \\
w_{17} &= 0.6 \quad &(\text{Smart Home Hub}) \\
w_{18} &= 4.0 \quad &(\text{Gaming Console}) \\
w_{19} &= 0.2 \quad &(\text{Fitness Tracker}) \\
w_{20} &= 0.5 \quad &(\text{E-Reader}) \\
\end{align*}
\]

Shelf capacities $C_i$ (from capacity.csv):

\[
\begin{align*}
C_1 &= 5.0 \\
C_2 &= 7.0 \\
C_3 &= 6.0 \\
C_4 &= 8.0 \\
C_5 &= 5.5 \\
C_6 &= 9.0 \\
C_7 &= 6.5 \\
C_8 &= 7.5 \\
C_9 &= 8.2 \\
C_{10} &= 5.7 \\
\end{align*}
\]

---

**Variable Domains:**

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
\]

---

**Complete Model:**

\[
\begin{align*}
\max\ & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{20} w_j x_{ij} \leq C_i \qquad \forall i = 1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10,\ j = 1,\ldots,20 \\
\end{align*}
\]

Where all $v_j$, $w_j$, and $C_i$ are as listed above, and product and shelf indices correspond to the order in the original CSV files.