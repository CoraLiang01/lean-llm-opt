Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Storage areas (indexed by $i$, StorageID):  
  1: Capacity = 1083  
  2: Capacity = 1840  
  3: Capacity = 770  
  4: Capacity = 1299  
  5: Capacity = 1259  
  6: Capacity = 543  
  7: Capacity = 1831  
  8: Capacity = 855  
  9: Capacity = 619  
  10: Capacity = 637  
  11: Capacity = 935  
  12: Capacity = 626  
  13: Capacity = 1457  
  14: Capacity = 1198  
  15: Capacity = 837  

- Air conditioner types (indexed by $j$, ProductName), with value $v_j$ and size $w_j$:

| ProductName           | Value ($v_j$) | Weight ($w_j$) |
|-----------------------|--------------|---------------|
| Window Unit           | 4811         | 114           |
| Portable Unit         | 1130         | 200           |
| Split System          | 1611         | 106           |
| Ductless System       | 3368         | 256           |
| Central AC            | 2135         | 268           |
| Hybrid AC             | 1046         | 185           |
| Geothermal AC         | 4030         | 299           |
| Smart AC              | 3761         | 131           |
| Evaporative Cooler    | 3523         | 139           |
| Package Unit          | 1701         | 105           |

---

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i \in \{1,\ldots,15\}} \sum_{j \in \{\text{all ProductNames}\}} v_j \cdot x_{ij}
\]

**Subject to:**

For each storage area $i$ (StorageID):

\[
\sum_{j} w_j \cdot x_{ij} \leq \text{Capacity}_i \qquad \forall i \in \{1,\ldots,15\}
\]

Where the capacities are:

\[
\begin{align*}
\sum_{j} w_j x_{1j} &\leq 1083 \\
\sum_{j} w_j x_{2j} &\leq 1840 \\
\sum_{j} w_j x_{3j} &\leq 770 \\
\sum_{j} w_j x_{4j} &\leq 1299 \\
\sum_{j} w_j x_{5j} &\leq 1259 \\
\sum_{j} w_j x_{6j} &\leq 543 \\
\sum_{j} w_j x_{7j} &\leq 1831 \\
\sum_{j} w_j x_{8j} &\leq 855 \\
\sum_{j} w_j x_{9j} &\leq 619 \\
\sum_{j} w_j x_{10j} &\leq 637 \\
\sum_{j} w_j x_{11j} &\leq 935 \\
\sum_{j} w_j x_{12j} &\leq 626 \\
\sum_{j} w_j x_{13j} &\leq 1457 \\
\sum_{j} w_j x_{14j} &\leq 1198 \\
\sum_{j} w_j x_{15j} &\leq 837 \\
\end{align*}
\]

**Variable domains:**

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,15\},\ j \in \{\text{all ProductNames}\}
\]

---

**Where:**

- $x_{ij}$: Number of units of air conditioner type $j$ placed in storage area $i$ (integer, $\geq 0$)
- $v_j$: Value of air conditioner type $j$ (see table above)
- $w_j$: Weight (size) of air conditioner type $j$ (see table above)
- $\text{Capacity}_i$: Capacity of storage area $i$ (see list above)