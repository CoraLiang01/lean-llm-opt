Let $x_{ij}$ be the number of units of product $j$ (with item_name $j$) placed on shelf $i$ (with resource_id $i$). All variables are nonnegative integers.

**Parameters:**

- For each shelf $i$ (resource_id from capacity.csv), the capacity is $C_i$ (resource_capacity).
- For each product $j$ (item_name from products.csv), the value per unit is $v_j$ (item_value), and the weight per unit is $w_j$ (resource_requirement).

**Objective:**

$$
\max \sum_{i \in \{1,2,3,4,5,6,7,8,9,10\}} \sum_{j \in \{1,2,\ldots,20\}} v_j \cdot x_{ij}
$$

where $v_j$ is as follows (from products.csv, in source order):

\[
\begin{align*}
v_1 &= 50 \\
v_2 &= 70 \\
v_3 &= 30 \\
v_4 &= 60 \\
v_5 &= 80 \\
v_6 &= 90 \\
v_7 &= 40 \\
v_8 &= 100 \\
v_9 &= 55 \\
v_{10} &= 75 \\
v_{11} &= 65 \\
v_{12} &= 95 \\
v_{13} &= 45 \\
v_{14} &= 85 \\
v_{15} &= 70 \\
v_{16} &= 110 \\
v_{17} &= 50 \\
v_{18} &= 60 \\
v_{19} &= 120 \\
v_{20} &= 100 \\
\end{align*}
\]

**Constraints:**

For each shelf $i$ (resource_id from capacity.csv):

\[
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i
\]

where $w_j$ is as follows (from products.csv, in source order):

\[
\begin{align*}
w_1 &= 10 \\
w_2 &= 20 \\
w_3 &= 5 \\
w_4 &= 15 \\
w_5 &= 25 \\
w_6 &= 30 \\
w_7 &= 12 \\
w_8 &= 35 \\
w_9 &= 10 \\
w_{10} &= 20 \\
w_{11} &= 18 \\
w_{12} &= 28 \\
w_{13} &= 8 \\
w_{14} &= 22 \\
w_{15} &= 25 \\
w_{16} &= 40 \\
w_{17} &= 14 \\
w_{18} &= 16 \\
w_{19} &= 50 \\
w_{20} &= 30 \\
\end{align*}
\]

and $C_i$ is as follows (from capacity.csv, in source order):

\[
\begin{align*}
C_1 &= 500 \\
C_2 &= 700 \\
C_3 &= 600 \\
C_4 &= 800 \\
C_5 &= 550 \\
C_6 &= 900 \\
C_7 &= 650 \\
C_8 &= 750 \\
C_9 &= 820 \\
C_{10} &= 570 \\
\end{align*}
\]

**Variable domains:**

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
\]

**Summary of Sets and Indices:**

- $i$ indexes shelves (resource_id): $1,2,3,4,5,6,7,8,9,10$
- $j$ indexes products (item_name): $1,2,3,\ldots,20$

**Complete Model:**

\[
\begin{align*}
\max\ & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.}\quad & \sum_{j=1}^{20} w_j x_{ij} \leq C_i, \quad \forall i=1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i=1,\ldots,10,\ j=1,\ldots,20
\end{align*}
\]

with $v_j$, $w_j$, and $C_i$ as specified above.