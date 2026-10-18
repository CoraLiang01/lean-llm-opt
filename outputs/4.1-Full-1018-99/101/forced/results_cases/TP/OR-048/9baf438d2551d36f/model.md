##### Sets

Let $I = \{1,2,\ldots,15\}$ be the set of storage areas (StorageID in source order).  
Let $J = \{$Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit$\}$ be the set of air conditioner types (ProductName in source order).

##### Parameters

For each storage area $i \in I$ (source order):

- $C_i$ = capacity of storage area $i$:

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

For each product $j \in J$ (source order):

- $v_j$ = value per unit of product $j$
- $w_j$ = weight (size) per unit of product $j$

\[
\begin{array}{lll}
\text{ProductName} & v_j & w_j \\
\hline
\text{Window Unit} & 4811 & 114 \\
\text{Portable Unit} & 1130 & 200 \\
\text{Split System} & 1611 & 106 \\
\text{Ductless System} & 3368 & 256 \\
\text{Central AC} & 2135 & 268 \\
\text{Hybrid AC} & 1046 & 185 \\
\text{Geothermal AC} & 4030 & 299 \\
\text{Smart AC} & 3761 & 131 \\
\text{Evaporative Cooler} & 3523 & 139 \\
\text{Package Unit} & 1701 & 105 \\
\end{array}
\]

##### Decision Variables

For each $i \in I$, $j \in J$:

- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed in storage area $i$

##### Objective

\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

##### Constraints

For each storage area $i \in I$:

\[
\sum_{j \in J} w_j x_{ij} \leq C_i
\]

For all $i \in I$, $j \in J$:

\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

---

###### Retrieved Information

Storage areas (source order):

1: 1083  
2: 1840  
3: 770  
4: 1299  
5: 1259  
6: 543  
7: 1831  
8: 855  
9: 619  
10: 637  
11: 935  
12: 626  
13: 1457  
14: 1198  
15: 837  

Products (source order):

- Window Unit: value 4811, weight 114
- Portable Unit: value 1130, weight 200
- Split System: value 1611, weight 106
- Ductless System: value 3368, weight 256
- Central AC: value 2135, weight 268
- Hybrid AC: value 1046, weight 185
- Geothermal AC: value 4030, weight 299
- Smart AC: value 3761, weight 131
- Evaporative Cooler: value 3523, weight 139
- Package Unit: value 1701, weight 105