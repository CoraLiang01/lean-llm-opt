##### Sets and Indices

Let $i$ index the products, with the following mapping (in source order):

1. Spinach  
2. Shiitake Mushrooms  
3. Apples  
4. Carrots  
5. Basil  
6. Potatoes  
7. Green Beans  
8. Blueberries  
9. Oranges  
10. Watermelons  

##### Parameters

- $v_i$: Value per unit of product $i$  
  Spinach: $64$  
  Shiitake Mushrooms: $75$  
  Apples: $68$  
  Carrots: $11$  
  Basil: $91$  
  Potatoes: $31$  
  Green Beans: $90$  
  Blueberries: $56$  
  Oranges: $10$  
  Watermelons: $24$  

- $w_i$: Weight per unit of product $i$  
  Spinach: $230$  
  Shiitake Mushrooms: $637$  
  Apples: $773$  
  Carrots: $653$  
  Basil: $755$  
  Potatoes: $670$  
  Green Beans: $505$  
  Blueberries: $821$  
  Oranges: $83$  
  Watermelons: $249$  

- $C$: Total stock capacity $= 875$

##### Decision Variables

- $x_i$: Number of units of product $i$ to order each day  
  Domain: $x_i \in \mathbb{Z}_{\geq 0}$ for all $i$

##### Objective Function

$$
\max \sum_{i=1}^{10} v_i x_i = 64x_1 + 75x_2 + 68x_3 + 11x_4 + 91x_5 + 31x_6 + 90x_7 + 56x_8 + 10x_9 + 24x_{10}
$$

##### Constraints

- Stock capacity constraint:
$$
230x_1 + 637x_2 + 773x_3 + 653x_4 + 755x_5 + 670x_6 + 505x_7 + 821x_8 + 83x_9 + 249x_{10} \leq 875
$$

- Non-negativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 10
$$

##### Complete Model

\[
\begin{align*}
\max \quad & 64x_1 + 75x_2 + 68x_3 + 11x_4 + 91x_5 + 31x_6 + 90x_7 + 56x_8 + 10x_9 + 24x_{10} \\
\text{s.t.} \quad & 230x_1 + 637x_2 + 773x_3 + 653x_4 + 755x_5 + 670x_6 + 505x_7 + 821x_8 + 83x_9 + 249x_{10} \leq 875 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 10
\end{align*}
\]

##### Retrieved Data

- Products (in source order):  
 Spinach, Shiitake Mushrooms, Apples, Carrots, Basil, Potatoes, Green Beans, Blueberries, Oranges, Watermelons

- Value per unit:  
 Spinach: 64, Shiitake Mushrooms: 75, Apples: 68, Carrots: 11, Basil: 91, Potatoes: 31, Green Beans: 90, Blueberries: 56, Oranges: 10, Watermelons: 24

- Weight per unit:  
 Spinach: 230, Shiitake Mushrooms: 637, Apples: 773, Carrots: 653, Basil: 755, Potatoes: 670, Green Beans: 505, Blueberries: 821, Oranges: 83, Watermelons: 249

- Total stock capacity: 875