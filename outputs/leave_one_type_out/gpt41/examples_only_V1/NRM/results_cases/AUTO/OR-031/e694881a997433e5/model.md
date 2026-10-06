Let $x_i$ denote the number of units of product $i$ to be fulfilled, for each product $i$ in the set below.

##### Sets and Parameters

Let $I$ be the set of all products, indexed by $i$:

\[
\begin{array}{ll}
\text{Butter\_Amul} & \text{Butter\_Mother Dairy} \\
\text{Butter\_Parag Milk Foods} & \text{Butter\_Warana} \\
\text{Buttermilk\_Amul} & \text{Buttermilk\_Mother Dairy} \\
\text{Buttermilk\_Raj} & \text{Buttermilk\_Sudha} \\
\text{Cheese\_Amul} & \text{Cheese\_Britannia Industries} \\
\text{Cheese\_Dynamix Dairies} & \text{Cheese\_Passion Cheese} \\
\text{Curd\_Amul} & \text{Curd\_Mother Dairy} \\
\text{Curd\_Raj} & \text{Curd\_Sudha} \\
\text{Ghee\_Amul} & \text{Ghee\_Mother Dairy} \\
\text{Ghee\_Raj} & \text{Ghee\_Sudha} \\
\text{Ice Cream\_Amul} & \text{Ice Cream\_Dodla Dairy} \\
\text{Ice Cream\_Mother Dairy} & \text{Ice Cream\_Palle2patnam} \\
\text{Lassi\_Amul} & \text{Lassi\_Mother Dairy} \\
\text{Lassi\_Raj} & \text{Lassi\_Sudha} \\
\text{Milk\_Amul} & \text{Milk\_Mother Dairy} \\
\text{Milk\_Raj} & \text{Milk\_Sudha} \\
\text{Paneer\_Amul} & \text{Paneer\_Mother Dairy} \\
\text{Paneer\_Raj} & \text{Paneer\_Sudha} \\
\text{Yogurt\_Amul} & \text{Yogurt\_Dodla Dairy} \\
\text{Yogurt\_Mother Dairy} & \text{Yogurt\_Palle2patnam} \\
\end{array}
\]

For each product $i$:
- $r_i$ = Revenue per unit (see table below)
- $d_i$ = Demand
- $s_i$ = Initial Inventory

\[
\begin{array}{lccc}
\text{Product} & r_i & d_i & s_i \\
\hline
\text{Butter\_Amul} & 96.86 & 34102 & 29862 \\
\text{Butter\_Mother Dairy} & 48.01 & 36579 & 29898 \\
\text{Butter\_Parag Milk Foods} & 8.83 & 36086 & 25208 \\
\text{Butter\_Warana} & 92.96 & 41254 & 30816 \\
\text{Buttermilk\_Amul} & 40.75 & 29876 & 19925 \\
\text{Buttermilk\_Mother Dairy} & 83.07 & 41229 & 26482 \\
\text{Buttermilk\_Raj} & 15.64 & 35354 & 30865 \\
\text{Buttermilk\_Sudha} & 56.57 & 29649 & 33517 \\
\text{Cheese\_Amul} & 100.74 & 38558 & 30929 \\
\text{Cheese\_Britannia Industries} & 28.92 & 28603 & 21405 \\
\text{Cheese\_Dynamix Dairies} & 32.66 & 35962 & 25953 \\
\text{Cheese\_Passion Cheese} & 58.09 & 36961 & 23825 \\
\text{Curd\_Amul} & 30.27 & 39436 & 31687 \\
\text{Curd\_Mother Dairy} & 84.57 & 43522 & 33377 \\
\text{Curd\_Raj} & 84.75 & 38128 & 34914 \\
\text{Curd\_Sudha} & 76.37 & 42341 & 33547 \\
\text{Ghee\_Amul} & 41.49 & 30345 & 23120 \\
\text{Ghee\_Mother Dairy} & 52.79 & 35420 & 24667 \\
\text{Ghee\_Raj} & 48.13 & 34100 & 25395 \\
\text{Ghee\_Sudha} & 95.09 & 33007 & 24676 \\
\text{Ice Cream\_Amul} & 54.41 & 37894 & 26707 \\
\text{Ice Cream\_Dodla Dairy} & 82.24 & 29840 & 26722 \\
\text{Ice Cream\_Mother Dairy} & 94.32 & 38762 & 25809 \\
\text{Ice Cream\_Palle2patnam} & 83.73 & 34674 & 24391 \\
\text{Lassi\_Amul} & 74.45 & 42972 & 30728 \\
\text{Lassi\_Mother Dairy} & 49.4 & 33894 & 28628 \\
\text{Lassi\_Raj} & 93.93 & 45762 & 30568 \\
\text{Lassi\_Sudha} & 88.05 & 29503 & 23461 \\
\text{Milk\_Amul} & 39.24 & 34761 & 21398 \\
\text{Milk\_Mother Dairy} & 8.69 & 40548 & 33619 \\
\text{Milk\_Raj} & 65.53 & 43012 & 26355 \\
\text{Milk\_Sudha} & 42.34 & 29180 & 23815 \\
\text{Paneer\_Amul} & 81.76 & 33498 & 20787 \\
\text{Paneer\_Mother Dairy} & 29.09 & 34848 & 29342 \\
\text{Paneer\_Raj} & 87.3 & 40347 & 23556 \\
\text{Paneer\_Sudha} & 66.7 & 37188 & 28753 \\
\text{Yogurt\_Amul} & 89.32 & 34347 & 24404 \\
\text{Yogurt\_Dodla Dairy} & 33.81 & 37181 & 26829 \\
\text{Yogurt\_Mother Dairy} & 25.29 & 36644 & 25562 \\
\text{Yogurt\_Palle2patnam} & 84.9 & 34303 & 28695 \\
\end{array}
\]

##### Decision Variables

For each $i \in I$:
- $x_i$ = number of units of product $i$ to fulfill (integer, $x_i \geq 0$)

##### Objective

\[
\max \sum_{i \in I} r_i x_i
\]

##### Constraints

For each $i \in I$:
- Inventory limit: $x_i \leq s_i$
- Demand limit: $x_i \leq d_i$
- Nonnegativity and integrality: $x_i \in \mathbb{Z}_{\geq 0}$

##### Complete Model

\[
\begin{align*}
\max \quad & \sum_{i \in I} r_i x_i \\
\text{s.t.} \quad & x_i \leq s_i, \quad \forall i \in I \\
                  & x_i \leq d_i, \quad \forall i \in I \\
                  & x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I \\
\end{align*}
\]

where all parameters and indices are as defined above, using the exact product names and coefficients from the data.