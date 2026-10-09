Let $x_{ij}$ be the number of units of air conditioner type $j$ to be placed in storage area $i$. All $x_{ij}$ are required to be nonnegative integers.

Define:
- $i \in \{\text{1}, \text{2}, \ldots, \text{15}\}$ (StorageID from capacity.csv)
- $j \in \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\}$ (ProductName from products.csv)

Let $v_j$ be the Value of product $j$, and $w_j$ be the Weight (size) of product $j$.

Let $C_i$ be the Capacity of storage area $i$.

#### Objective Function

$$
\max \sum_{i \in \{1,\ldots,15\}} \sum_{j \in \{\text{Window Unit}, \ldots, \text{Package Unit}\}} v_j \cdot x_{ij}
$$

where

\[
\begin{align*}
v_{\text{Window Unit}} &= 4811 \\
v_{\text{Portable Unit}} &= 1130 \\
v_{\text{Split System}} &= 1611 \\
v_{\text{Ductless System}} &= 3368 \\
v_{\text{Central AC}} &= 2135 \\
v_{\text{Hybrid AC}} &= 1046 \\
v_{\text{Geothermal AC}} &= 4030 \\
v_{\text{Smart AC}} &= 3761 \\
v_{\text{Evaporative Cooler}} &= 3523 \\
v_{\text{Package Unit}} &= 1701 \\
\end{align*}
\]

#### Constraints

For each storage area $i$ (StorageID):

$$
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,15\}
$$

where

\[
\begin{align*}
C_1 &= 1083 \\
C_2 &= 1840 \\
C_3 &= 770 \\
C_4 &= 1299 \\
C_5 &= 1259 \\
C_6 &= 543 \\
C_7 &= 1831 \\
C_8 &= 855 \\
C_9 &= 619 \\
C_{10} &= 637 \\
C_{11} &= 935 \\
C_{12} &= 626 \\
C_{13} &= 1457 \\
C_{14} &= 1198 \\
C_{15} &= 837 \\
\end{align*}
\]

and

\[
\begin{align*}
w_{\text{Window Unit}} &= 114 \\
w_{\text{Portable Unit}} &= 200 \\
w_{\text{Split System}} &= 106 \\
w_{\text{Ductless System}} &= 256 \\
w_{\text{Central AC}} &= 268 \\
w_{\text{Hybrid AC}} &= 185 \\
w_{\text{Geothermal AC}} &= 299 \\
w_{\text{Smart AC}} &= 131 \\
w_{\text{Evaporative Cooler}} &= 139 \\
w_{\text{Package Unit}} &= 105 \\
\end{align*}
\]

#### Variable Domains

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

#### Summary of Sets and Parameters

- Storage areas (StorageID): 1, 2, ..., 15
- Air conditioner types (ProductName): Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit
- $v_j$: Value of each product (see above)
- $w_j$: Weight (size) of each product (see above)
- $C_i$: Capacity of each storage area (see above)

#### Complete Model

$$
\begin{align*}
\max \quad & \sum_{i=1}^{15} \sum_{j=1}^{10} v_j x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{10} w_j x_{ij} \leq C_i \qquad \forall i=1,\ldots,15 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i=1,\ldots,15;\ j=1,\ldots,10
\end{align*}
$$

with all coefficients and identifiers as listed above.