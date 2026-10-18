Let $x_i$ denote the number of units of product $i$ to be fulfilled, for each product $i$ in the set below.

**Sets and Parameters (from source, in order):**

| Full_Product_Name                | Revenue | Demand | Initial Inventory |
|----------------------------------|---------|--------|------------------|
| Butter_Amul                      | 96.86   | 34102  | 29862            |
| Butter_Mother Dairy              | 48.01   | 36579  | 29898            |
| Butter_Parag Milk Foods          | 8.83    | 36086  | 25208            |
| Butter_Warana                    | 92.96   | 41254  | 30816            |
| Buttermilk_Amul                  | 40.75   | 29876  | 19925            |
| Buttermilk_Mother Dairy          | 83.07   | 41229  | 26482            |
| Buttermilk_Raj                   | 15.64   | 35354  | 30865            |
| Buttermilk_Sudha                 | 56.57   | 29649  | 33517            |
| Cheese_Amul                      | 100.74  | 38558  | 30929            |
| Cheese_Britannia Industries      | 28.92   | 28603  | 21405            |
| Cheese_Dynamix Dairies           | 32.66   | 35962  | 25953            |
| Cheese_Passion Cheese            | 58.09   | 36961  | 23825            |
| Curd_Amul                        | 30.27   | 39436  | 31687            |
| Curd_Mother Dairy                | 84.57   | 43522  | 33377            |
| Curd_Raj                         | 84.75   | 38128  | 34914            |
| Curd_Sudha                       | 76.37   | 42341  | 33547            |
| Ghee_Amul                        | 41.49   | 30345  | 23120            |
| Ghee_Mother Dairy                | 52.79   | 35420  | 24667            |
| Ghee_Raj                         | 48.13   | 34100  | 25395            |
| Ghee_Sudha                       | 95.09   | 33007  | 24676            |
| Ice Cream_Amul                   | 54.41   | 37894  | 26707            |
| Ice Cream_Dodla Dairy            | 82.24   | 29840  | 26722            |
| Ice Cream_Mother Dairy           | 94.32   | 38762  | 25809            |
| Ice Cream_Palle2patnam           | 83.73   | 34674  | 24391            |
| Lassi_Amul                       | 74.45   | 42972  | 30728            |
| Lassi_Mother Dairy               | 49.4    | 33894  | 28628            |
| Lassi_Raj                        | 93.93   | 45762  | 30568            |
| Lassi_Sudha                      | 88.05   | 29503  | 23461            |
| Milk_Amul                        | 39.24   | 34761  | 21398            |
| Milk_Mother Dairy                | 8.69    | 40548  | 33619            |
| Milk_Raj                         | 65.53   | 43012  | 26355            |
| Milk_Sudha                       | 42.34   | 29180  | 23815            |
| Paneer_Amul                      | 81.76   | 33498  | 20787            |
| Paneer_Mother Dairy              | 29.09   | 34848  | 29342            |
| Paneer_Raj                       | 87.3    | 40347  | 23556            |
| Paneer_Sudha                     | 66.7    | 37188  | 28753            |
| Yogurt_Amul                      | 89.32   | 34347  | 24404            |
| Yogurt_Dodla Dairy               | 33.81   | 37181  | 26829            |
| Yogurt_Mother Dairy              | 25.29   | 36644  | 25562            |
| Yogurt_Palle2patnam              | 84.9    | 34303  | 28695            |

**Mathematical Model:**

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$, for each product $i$ (see table above): number of units of product $i$ to fulfill.

**Objective:**
\[
\max \sum_{i} \text{Revenue}_i \cdot x_i
\]
where $\text{Revenue}_i$ is the per-unit revenue for product $i$.

**Constraints:**
- Inventory and demand limits for each product $i$:
  \[
  0 \leq x_i \leq \min\{\text{Demand}_i, \text{Initial Inventory}_i\} \qquad \forall i
  \]
  where $\text{Demand}_i$ and $\text{Initial Inventory}_i$ are as in the table above.

- Integrality:
  \[
  x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i
  \]

**Explicitly, for each product $i$ (in source order):**
\[
\begin{align*}
0 &\leq x_{\text{Butter\_Amul}} \leq 29862 \\
0 &\leq x_{\text{Butter\_Mother Dairy}} \leq 29898 \\
0 &\leq x_{\text{Butter\_Parag Milk Foods}} \leq 25208 \\
0 &\leq x_{\text{Butter\_Warana}} \leq 30816 \\
0 &\leq x_{\text{Buttermilk\_Amul}} \leq 19925 \\
0 &\leq x_{\text{Buttermilk\_Mother Dairy}} \leq 26482 \\
0 &\leq x_{\text{Buttermilk\_Raj}} \leq 30865 \\
0 &\leq x_{\text{Buttermilk\_Sudha}} \leq 29649 \\
0 &\leq x_{\text{Cheese\_Amul}} \leq 30929 \\
0 &\leq x_{\text{Cheese\_Britannia Industries}} \leq 21405 \\
0 &\leq x_{\text{Cheese\_Dynamix Dairies}} \leq 25953 \\
0 &\leq x_{\text{Cheese\_Passion Cheese}} \leq 23825 \\
0 &\leq x_{\text{Curd\_Amul}} \leq 31687 \\
0 &\leq x_{\text{Curd\_Mother Dairy}} \leq 33377 \\
0 &\leq x_{\text{Curd\_Raj}} \leq 34914 \\
0 &\leq x_{\text{Curd\_Sudha}} \leq 33547 \\
0 &\leq x_{\text{Ghee\_Amul}} \leq 23120 \\
0 &\leq x_{\text{Ghee\_Mother Dairy}} \leq 24667 \\
0 &\leq x_{\text{Ghee\_Raj}} \leq 25395 \\
0 &\leq x_{\text{Ghee\_Sudha}} \leq 24676 \\
0 &\leq x_{\text{Ice Cream\_Amul}} \leq 26707 \\
0 &\leq x_{\text{Ice Cream\_Dodla Dairy}} \leq 26722 \\
0 &\leq x_{\text{Ice Cream\_Mother Dairy}} \leq 25809 \\
0 &\leq x_{\text{Ice Cream\_Palle2patnam}} \leq 24391 \\
0 &\leq x_{\text{Lassi\_Amul}} \leq 30728 \\
0 &\leq x_{\text{Lassi\_Mother Dairy}} \leq 28628 \\
0 &\leq x_{\text{Lassi\_Raj}} \leq 30568 \\
0 &\leq x_{\text{Lassi\_Sudha}} \leq 23461 \\
0 &\leq x_{\text{Milk\_Amul}} \leq 21398 \\
0 &\leq x_{\text{Milk\_Mother Dairy}} \leq 33619 \\
0 &\leq x_{\text{Milk\_Raj}} \leq 26355 \\
0 &\leq x_{\text{Milk\_Sudha}} \leq 23815 \\
0 &\leq x_{\text{Paneer\_Amul}} \leq 20787 \\
0 &\leq x_{\text{Paneer\_Mother Dairy}} \leq 29342 \\
0 &\leq x_{\text{Paneer\_Raj}} \leq 23556 \\
0 &\leq x_{\text{Paneer\_Sudha}} \leq 28753 \\
0 &\leq x_{\text{Yogurt\_Amul}} \leq 24404 \\
0 &\leq x_{\text{Yogurt\_Dodla Dairy}} \leq 26829 \\
0 &\leq x_{\text{Yogurt\_Mother Dairy}} \leq 25562 \\
0 &\leq x_{\text{Yogurt\_Palle2patnam}} \leq 28695 \\
\end{align*}
\]

**Variable domains:**
\[
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i
\]

**Summary:**  
Maximize total revenue from fulfilled orders, subject to not exceeding either demand or initial inventory for each product, with all variables nonnegative integers. All coefficients and bounds are as retrieved above, in source order.