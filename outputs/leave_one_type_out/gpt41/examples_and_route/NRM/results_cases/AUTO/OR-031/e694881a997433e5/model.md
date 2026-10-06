Let $i$ index the dairy products, with identifiers as in the Full_Product_Name column.

Decision variables:
$$
x_i = \text{number of units of product } i \text{ to be fulfilled}, \quad x_i \in \mathbb{Z}_{\geq 0}
$$

Parameters (for each product $i$):

- $r_i$ = Revenue per unit (from Revenue column)
- $d_i$ = Demand (from Demand column)
- $s_i$ = Initial Inventory (from Initial Inventory column)

Objective:
$$
\max \sum_i r_i x_i
$$

Subject to:

- Demand fulfillment: $x_i \leq d_i \quad \forall i$
- Inventory limit:   $x_i \leq s_i \quad \forall i$
- Nonnegativity and integrality: $x_i \in \mathbb{Z}_{\geq 0} \quad \forall i$

Numerical Formulation (all products in source order):

Maximize
$$
96.86\,x_{\text{Butter\_Amul}} + 48.01\,x_{\text{Butter\_Mother Dairy}} + 8.83\,x_{\text{Butter\_Parag Milk Foods}} + 92.96\,x_{\text{Butter\_Warana}} + 40.75\,x_{\text{Buttermilk\_Amul}} + 83.07\,x_{\text{Buttermilk\_Mother Dairy}} + 15.64\,x_{\text{Buttermilk\_Raj}} + 56.57\,x_{\text{Buttermilk\_Sudha}} + 100.74\,x_{\text{Cheese\_Amul}} + 28.92\,x_{\text{Cheese\_Britannia Industries}} + 32.66\,x_{\text{Cheese\_Dynamix Dairies}} + 58.09\,x_{\text{Cheese\_Passion Cheese}} + 30.27\,x_{\text{Curd\_Amul}} + 84.57\,x_{\text{Curd\_Mother Dairy}} + 84.75\,x_{\text{Curd\_Raj}} + 76.37\,x_{\text{Curd\_Sudha}} + 41.49\,x_{\text{Ghee\_Amul}} + 52.79\,x_{\text{Ghee\_Mother Dairy}} + 48.13\,x_{\text{Ghee\_Raj}} + 95.09\,x_{\text{Ghee\_Sudha}} + 54.41\,x_{\text{Ice Cream\_Amul}} + 82.24\,x_{\text{Ice Cream\_Dodla Dairy}} + 94.32\,x_{\text{Ice Cream\_Mother Dairy}} + 83.73\,x_{\text{Ice Cream\_Palle2patnam}} + 74.45\,x_{\text{Lassi\_Amul}} + 49.4\,x_{\text{Lassi\_Mother Dairy}} + 93.93\,x_{\text{Lassi\_Raj}} + 88.05\,x_{\text{Lassi\_Sudha}} + 39.24\,x_{\text{Milk\_Amul}} + 8.69\,x_{\text{Milk\_Mother Dairy}} + 65.53\,x_{\text{Milk\_Raj}} + 42.34\,x_{\text{Milk\_Sudha}} + 81.76\,x_{\text{Paneer\_Amul}} + 29.09\,x_{\text{Paneer\_Mother Dairy}} + 87.3\,x_{\text{Paneer\_Raj}} + 66.7\,x_{\text{Paneer\_Sudha}} + 89.32\,x_{\text{Yogurt\_Amul}} + 33.81\,x_{\text{Yogurt\_Dodla Dairy}} + 25.29\,x_{\text{Yogurt\_Mother Dairy}} + 84.9\,x_{\text{Yogurt\_Palle2patnam}}
$$

Subject to, for each product $i$ (in source order):

\[
\begin{align*}
x_{\text{Butter\_Amul}} &\leq 34102 \\
x_{\text{Butter\_Amul}} &\leq 29862 \\
x_{\text{Butter\_Mother Dairy}} &\leq 36579 \\
x_{\text{Butter\_Mother Dairy}} &\leq 29898 \\
x_{\text{Butter\_Parag Milk Foods}} &\leq 36086 \\
x_{\text{Butter\_Parag Milk Foods}} &\leq 25208 \\
x_{\text{Butter\_Warana}} &\leq 41254 \\
x_{\text{Butter\_Warana}} &\leq 30816 \\
x_{\text{Buttermilk\_Amul}} &\leq 29876 \\
x_{\text{Buttermilk\_Amul}} &\leq 19925 \\
x_{\text{Buttermilk\_Mother Dairy}} &\leq 41229 \\
x_{\text{Buttermilk\_Mother Dairy}} &\leq 26482 \\
x_{\text{Buttermilk\_Raj}} &\leq 35354 \\
x_{\text{Buttermilk\_Raj}} &\leq 30865 \\
x_{\text{Buttermilk\_Sudha}} &\leq 29649 \\
x_{\text{Buttermilk\_Sudha}} &\leq 33517 \\
x_{\text{Cheese\_Amul}} &\leq 38558 \\
x_{\text{Cheese\_Amul}} &\leq 30929 \\
x_{\text{Cheese\_Britannia Industries}} &\leq 28603 \\
x_{\text{Cheese\_Britannia Industries}} &\leq 21405 \\
x_{\text{Cheese\_Dynamix Dairies}} &\leq 35962 \\
x_{\text{Cheese\_Dynamix Dairies}} &\leq 25953 \\
x_{\text{Cheese\_Passion Cheese}} &\leq 36961 \\
x_{\text{Cheese\_Passion Cheese}} &\leq 23825 \\
x_{\text{Curd\_Amul}} &\leq 39436 \\
x_{\text{Curd\_Amul}} &\leq 31687 \\
x_{\text{Curd\_Mother Dairy}} &\leq 43522 \\
x_{\text{Curd\_Mother Dairy}} &\leq 33377 \\
x_{\text{Curd\_Raj}} &\leq 38128 \\
x_{\text{Curd\_Raj}} &\leq 34914 \\
x_{\text{Curd\_Sudha}} &\leq 42341 \\
x_{\text{Curd\_Sudha}} &\leq 33547 \\
x_{\text{Ghee\_Amul}} &\leq 30345 \\
x_{\text{Ghee\_Amul}} &\leq 23120 \\
x_{\text{Ghee\_Mother Dairy}} &\leq 35420 \\
x_{\text{Ghee\_Mother Dairy}} &\leq 24667 \\
x_{\text{Ghee\_Raj}} &\leq 34100 \\
x_{\text{Ghee\_Raj}} &\leq 25395 \\
x_{\text{Ghee\_Sudha}} &\leq 33007 \\
x_{\text{Ghee\_Sudha}} &\leq 24676 \\
x_{\text{Ice Cream\_Amul}} &\leq 37894 \\
x_{\text{Ice Cream\_Amul}} &\leq 26707 \\
x_{\text{Ice Cream\_Dodla Dairy}} &\leq 29840 \\
x_{\text{Ice Cream\_Dodla Dairy}} &\leq 26722 \\
x_{\text{Ice Cream\_Mother Dairy}} &\leq 38762 \\
x_{\text{Ice Cream\_Mother Dairy}} &\leq 25809 \\
x_{\text{Ice Cream\_Palle2patnam}} &\leq 34674 \\
x_{\text{Ice Cream\_Palle2patnam}} &\leq 24391 \\
x_{\text{Lassi\_Amul}} &\leq 42972 \\
x_{\text{Lassi\_Amul}} &\leq 30728 \\
x_{\text{Lassi\_Mother Dairy}} &\leq 33894 \\
x_{\text{Lassi\_Mother Dairy}} &\leq 28628 \\
x_{\text{Lassi\_Raj}} &\leq 45762 \\
x_{\text{Lassi\_Raj}} &\leq 30568 \\
x_{\text{Lassi\_Sudha}} &\leq 29503 \\
x_{\text{Lassi\_Sudha}} &\leq 23461 \\
x_{\text{Milk\_Amul}} &\leq 34761 \\
x_{\text{Milk\_Amul}} &\leq 21398 \\
x_{\text{Milk\_Mother Dairy}} &\leq 40548 \\
x_{\text{Milk\_Mother Dairy}} &\leq 33619 \\
x_{\text{Milk\_Raj}} &\leq 43012 \\
x_{\text{Milk\_Raj}} &\leq 26355 \\
x_{\text{Milk\_Sudha}} &\leq 29180 \\
x_{\text{Milk\_Sudha}} &\leq 23815 \\
x_{\text{Paneer\_Amul}} &\leq 33498 \\
x_{\text{Paneer\_Amul}} &\leq 20787 \\
x_{\text{Paneer\_Mother Dairy}} &\leq 34848 \\
x_{\text{Paneer\_Mother Dairy}} &\leq 29342 \\
x_{\text{Paneer\_Raj}} &\leq 40347 \\
x_{\text{Paneer\_Raj}} &\leq 23556 \\
x_{\text{Paneer\_Sudha}} &\leq 37188 \\
x_{\text{Paneer\_Sudha}} &\leq 28753 \\
x_{\text{Yogurt\_Amul}} &\leq 34347 \\
x_{\text{Yogurt\_Amul}} &\leq 24404 \\
x_{\text{Yogurt\_Dodla Dairy}} &\leq 37181 \\
x_{\text{Yogurt\_Dodla Dairy}} &\leq 26829 \\
x_{\text{Yogurt\_Mother Dairy}} &\leq 36644 \\
x_{\text{Yogurt\_Mother Dairy}} &\leq 25562 \\
x_{\text{Yogurt\_Palle2patnam}} &\leq 34303 \\
x_{\text{Yogurt\_Palle2patnam}} &\leq 28695 \\
\end{align*}
\]

and

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
$$

where for each $i$, $x_i$ is indexed by the Full_Product_Name as above.