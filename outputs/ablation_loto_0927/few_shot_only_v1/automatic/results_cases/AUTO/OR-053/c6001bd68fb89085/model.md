**Mathematical Optimization Model**

Let  
- \( x_{ij} \): number of units of product \( j \) (ProductName from products.csv) to be placed on shelf \( i \) (ShelfID from capacity.csv), for all \( i \in \{1,2,\ldots,10\} \), \( j \in \{1,2,\ldots,20\} \).  
- \( v_j \): value of product \( j \) (Value from products.csv)  
- \( w_j \): weight of product \( j \) (Weight from products.csv)  
- \( C_i \): capacity of shelf \( i \) (Capacity from capacity.csv)  

**Indices:**  
- \( i \): ShelfID, \( i \in \{1,2,\ldots,10\} \)  
- \( j \): ProductName, \( j \in \{1,2,\ldots,20\} \)  

---

### Objective

\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
\]

where the values \( v_j \) are:

\[
\begin{align*}
v_1 &= 50 & v_{11} &= 65 \\
v_2 &= 70 & v_{12} &= 95 \\
v_3 &= 30 & v_{13} &= 45 \\
v_4 &= 60 & v_{14} &= 85 \\
v_5 &= 80 & v_{15} &= 70 \\
v_6 &= 90 & v_{16} &= 110 \\
v_7 &= 40 & v_{17} &= 50 \\
v_8 &= 100 & v_{18} &= 60 \\
v_9 &= 55 & v_{19} &= 120 \\
v_{10} &= 75 & v_{20} &= 100 \\
\end{align*}
\]

---

### Constraints

#### 1. Shelf Capacity Constraints

For each shelf \( i \in \{1,2,\ldots,10\} \):

\[
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i
\]

where the weights \( w_j \) and shelf capacities \( C_i \) are:

\[
\begin{align*}
w_1 &= 10 & C_1 &= 500 \\
w_2 &= 20 & C_2 &= 700 \\
w_3 &= 5 & C_3 &= 600 \\
w_4 &= 15 & C_4 &= 800 \\
w_5 &= 25 & C_5 &= 550 \\
w_6 &= 30 & C_6 &= 900 \\
w_7 &= 12 & C_7 &= 650 \\
w_8 &= 35 & C_8 &= 750 \\
w_9 &= 10 & C_9 &= 820 \\
w_{10} &= 20 & C_{10} &= 570 \\
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

#### 2. Nonnegativity and Integrality

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
\]

---

### Complete Model (Numerical Form)

\[
\begin{align*}
\max\ & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.}\quad
& \sum_{j=1}^{20} w_j x_{i j} \leq C_i \qquad \forall i = 1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10,\ j = 1,\ldots,20
\end{align*}
\]

where

- \( v_j \) and \( w_j \) as above for \( j = 1,\ldots,20 \)
- \( C_i \) as above for \( i = 1,\ldots,10 \)