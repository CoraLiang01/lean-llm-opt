Let:
- \( S = \{1,2,3,4,5,6,7,8,9,10\} \) be the set of shelves, indexed by ShelfID as given in capacity.csv.
- \( P \) be the set of products, indexed in the order given in products.csv, with names as ProductName.

Define decision variables:
- \( x_{ij} \): integer, number of units of product \( j \) (ProductName) to be placed on shelf \( i \) (ShelfID), for all \( i \in S \), \( j \in P \).
- \( x_{ij} \geq 0 \), integer.

Parameters (from the data):

From capacity.csv (in order):
\[
\begin{array}{ll}
\text{ShelfID} & \text{Capacity} \\
1 & 5.0 \\
2 & 7.0 \\
3 & 6.0 \\
4 & 8.0 \\
5 & 5.5 \\
6 & 9.0 \\
7 & 6.5 \\
8 & 7.5 \\
9 & 8.2 \\
10 & 5.7 \\
\end{array}
\]

From products.csv (in order):
\[
\begin{array}{lll}
\text{ProductName} & \text{Value} & \text{Weight} \\
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

Model:

\[
\begin{align*}
\text{Maximize} \quad & \sum_{i \in S} \sum_{j \in P} \text{Value}_j \cdot x_{ij} \\
\text{Subject to:} \quad & \sum_{j \in P} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i \quad \forall i \in S \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S, j \in P
\end{align*}
\]

Where:
- \( \text{Value}_j \) and \( \text{Weight}_j \) are as listed above for each ProductName \( j \).
- \( \text{Capacity}_i \) is as listed above for each ShelfID \( i \).

Explicitly, for each shelf \( i \) (from 1 to 10), the capacity constraint is:
\[
\sum_{j=1}^{20} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
\]
with the weights and capacities as given in the tables above, and all \( x_{ij} \) integer and nonnegative.

Objective:
\[
\text{Maximize} \quad \sum_{i=1}^{10} \sum_{j=1}^{20} \text{Value}_j \cdot x_{ij}
\]

Decision variables:
\[
x_{ij} \in \{0,1,2,\ldots\} \quad \forall i=1,\ldots,10; \; j=1,\ldots,20
\]

This model maximizes the total value of products allocated to shelves, subject to each shelf's weight capacity, using the explicit data and indices as provided.