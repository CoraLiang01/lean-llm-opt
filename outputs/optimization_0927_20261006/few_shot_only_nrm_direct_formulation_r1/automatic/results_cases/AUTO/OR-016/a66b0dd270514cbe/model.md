**Index Sets:**  
Let $I$ be the set of all products in the dataset, indexed by $i$.

$$
I = \{
\text{"Beauty - 25"},\ 
\text{"Beauty - 30"},\ 
\text{"Beauty - 300"},\ 
\text{"Beauty - 50"},\ 
\text{"Beauty - 500"},\ 
\text{"Clothing - 25"},\ 
\text{"Clothing - 30"},\ 
\text{"Clothing - 300"},\ 
\text{"Clothing - 50"},\ 
\text{"Clothing - 500"},\ 
\text{"Electronics - 25"},\ 
\text{"Electronics - 30"},\ 
\text{"Electronics - 300"},\ 
\text{"Electronics - 50"},\ 
\text{"Electronics - 500"},\ 
\text{"Home Goods - 25"},\ 
\text{"Home Goods - 30"},\ 
\text{"Home Goods - 50"},\ 
\text{"Home Goods - 300"},\ 
\text{"Home Goods - 500"},\ 
\text{"Sports - 25"},\ 
\text{"Sports - 30"},\ 
\text{"Sports - 50"},\ 
\text{"Sports - 300"},\ 
\text{"Sports - 500"},\ 
\text{"Furniture - 25"},\ 
\text{"Furniture - 30"},\ 
\text{"Furniture - 50"},\ 
\text{"Furniture - 300"},\ 
\text{"Furniture - 500"},\ 
\text{"Toys - 25"},\ 
\text{"Toys - 30"},\ 
\text{"Toys - 50"},\ 
\text{"Toys - 300"},\ 
\text{"Toys - 500"}
\}
$$

**Parameters:**  
For each $i \in I$:

- $A_i$: Revenue per unit of product $i$  
- $d_i$: Demand for product $i$  
- $I_i$: Initial inventory for product $i$  

Parameter values (in source order):

| $i$                           | $A_i$ | $d_i$ | $I_i$ |
|-------------------------------|-------|-------|-------|
| Beauty - 25                   | 25    | 240   | 1570  |
| Beauty - 30                   | 30    | 202   | 1330  |
| Beauty - 300                  | 300   | 216   | 1420  |
| Beauty - 50                   | 50    | 263   | 1700  |
| Beauty - 500                  | 500   | 256   | 1690  |
| Clothing - 25                 | 25    | 281   | 1840  |
| Clothing - 30                 | 30    | 261   | 1710  |
| Clothing - 300                | 300   | 295   | 1930  |
| Clothing - 50                 | 50    | 290   | 1890  |
| Clothing - 500                | 500   | 244   | 1570  |
| Electronics - 25              | 25    | 273   | 1810  |
| Electronics - 30              | 30    | 220   | 1410  |
| Electronics - 300             | 300   | 286   | 1830  |
| Electronics - 50              | 50    | 268   | 1750  |
| Electronics - 500             | 500   | 262   | 1690  |
| Home Goods - 25               | 25    | 255   | 1660  |
| Home Goods - 30               | 30    | 218   | 1417  |
| Home Goods - 50               | 50    | 278   | 1807  |
| Home Goods - 300              | 300   | 195   | 1268  |
| Home Goods - 500              | 500   | 248   | 1612  |
| Sports - 25                   | 25    | 269   | 1749  |
| Sports - 30                   | 30    | 227   | 1476  |
| Sports - 50                   | 50    | 285   | 1853  |
| Sports - 300                  | 300   | 208   | 1352  |
| Sports - 500                  | 500   | 259   | 1684  |
| Furniture - 25                | 25    | 242   | 1573  |
| Furniture - 30                | 30    | 213   | 1385  |
| Furniture - 50                | 50    | 272   | 1768  |
| Furniture - 300               | 300   | 224   | 1456  |
| Furniture - 500               | 500   | 251   | 1632  |
| Toys - 25                     | 25    | 257   | 1671  |
| Toys - 30                     | 30    | 235   | 1528  |
| Toys - 50                     | 50    | 291   | 1892  |
| Toys - 300                    | 300   | 199   | 1294  |
| Toys - 500                    | 500   | 264   | 1716  |

**Decision Variables:**  
For each $i \in I$:

- $x_i$: Number of units of product $i$ to fulfill  
  $x_i \in \mathbb{Z}_{+}$ (non-negative integer)

**Objective Function:**  
Maximize total revenue:

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**  

1. **Demand and Inventory Bounds:**  
   For all $i \in I$,
   $$
   0 \leq x_i \leq \min\{d_i, I_i\}
   $$

2. **Variable Domain:**  
   For all $i \in I$,
   $$
   x_i \in \mathbb{Z}_{+}
   $$

**Explicitly, for each $i$ in source order:**

- $0 \leq x_{\text{Beauty - 25}} \leq 240$
- $0 \leq x_{\text{Beauty - 30}} \leq 202$
- $0 \leq x_{\text{Beauty - 300}} \leq 216$
- $0 \leq x_{\text{Beauty - 50}} \leq 263$
- $0 \leq x_{\text{Beauty - 500}} \leq 256$
- $0 \leq x_{\text{Clothing - 25}} \leq 281$
- $0 \leq x_{\text{Clothing - 30}} \leq 261$
- $0 \leq x_{\text{Clothing - 300}} \leq 295$
- $0 \leq x_{\text{Clothing - 50}} \leq 290$
- $0 \leq x_{\text{Clothing - 500}} \leq 244$
- $0 \leq x_{\text{Electronics - 25}} \leq 273$
- $0 \leq x_{\text{Electronics - 30}} \leq 220$
- $0 \leq x_{\text{Electronics - 300}} \leq 286$
- $0 \leq x_{\text{Electronics - 50}} \leq 268$
- $0 \leq x_{\text{Electronics - 500}} \leq 262$
- $0 \leq x_{\text{Home Goods - 25}} \leq 255$
- $0 \leq x_{\text{Home Goods - 30}} \leq 218$
- $0 \leq x_{\text{Home Goods - 50}} \leq 278$
- $0 \leq x_{\text{Home Goods - 300}} \leq 195$
- $0 \leq x_{\text{Home Goods - 500}} \leq 248$
- $0 \leq x_{\text{Sports - 25}} \leq 269$
- $0 \leq x_{\text{Sports - 30}} \leq 227$
- $0 \leq x_{\text{Sports - 50}} \leq 285$
- $0 \leq x_{\text{Sports - 300}} \leq 208$
- $0 \leq x_{\text{Sports - 500}} \leq 259$
- $0 \leq x_{\text{Furniture - 25}} \leq 242$
- $0 \leq x_{\text{Furniture - 30}} \leq 213$
- $0 \leq x_{\text{Furniture - 50}} \leq 272$
- $0 \leq x_{\text{Furniture - 300}} \leq 224$
- $0 \leq x_{\text{Furniture - 500}} \leq 251$
- $0 \leq x_{\text{Toys - 25}} \leq 257$
- $0 \leq x_{\text{Toys - 30}} \leq 235$
- $0 \leq x_{\text{Toys - 50}} \leq 291$
- $0 \leq x_{\text{Toys - 300}} \leq 199$
- $0 \leq x_{\text{Toys - 500}} \leq 264$

**Summary:**  
Maximize
$$
\sum_{i \in I} A_i x_i
$$
subject to
$$
0 \leq x_i \leq \min\{d_i, I_i\},\quad x_i \in \mathbb{Z}_+,\quad \forall i \in I
$$

**All parameter values and product identifiers are as listed above, in source order.**