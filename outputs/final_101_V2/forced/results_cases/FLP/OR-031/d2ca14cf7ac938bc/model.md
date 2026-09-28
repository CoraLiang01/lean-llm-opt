##### Decision Variables

For each product $i \in P$ (where $P$ is the set of all products listed below):

$x_i \geq 0$: Number of units of product $i$ to fulfill (continuous, bounded above).

##### Parameters

Let $P$ be the set of products:

- Butter_Amul
- Butter_Mother Dairy
- Butter_Parag Milk Foods
- Butter_Warana
- Buttermilk_Amul
- Buttermilk_Mother Dairy
- Buttermilk_Raj
- Buttermilk_Sudha
- Cheese_Amul
- Cheese_Britannia Industries
- Cheese_Dynamix Dairies
- Cheese_Passion Cheese
- Curd_Amul
- Curd_Mother Dairy
- Curd_Raj
- Curd_Sudha
- Ghee_Amul
- Ghee_Mother Dairy
- Ghee_Raj
- Ghee_Sudha
- Ice Cream_Amul
- Ice Cream_Dodla Dairy
- Ice Cream_Mother Dairy
- Ice Cream_Palle2patnam
- Lassi_Amul
- Lassi_Mother Dairy
- Lassi_Raj
- Lassi_Sudha
- Milk_Amul
- Milk_Mother Dairy
- Milk_Raj
- Milk_Sudha
- Paneer_Amul
- Paneer_Mother Dairy
- Paneer_Raj
- Paneer_Sudha
- Yogurt_Amul
- Yogurt_Dodla Dairy
- Yogurt_Mother Dairy
- Yogurt_Palle2patnam

For each product $i \in P$:

- $I_i$: Initial Inventory
- $D_i$: Demand
- $r_i$: Revenue per unit

The parameter values are:

| Product                        | $I_i$  | $D_i$  | $r_i$   |
|--------------------------------|--------|--------|---------|
| Butter_Amul                    | 29862  | 34102  | 96.86   |
| Butter_Mother Dairy            | 29898  | 36579  | 48.01   |
| Butter_Parag Milk Foods        | 25208  | 36086  | 8.83    |
| Butter_Warana                  | 30816  | 41254  | 92.96   |
| Buttermilk_Amul                | 19925  | 29876  | 40.75   |
| Buttermilk_Mother Dairy        | 26482  | 41229  | 83.07   |
| Buttermilk_Raj                 | 30865  | 35354  | 15.64   |
| Buttermilk_Sudha               | 33517  | 29649  | 56.57   |
| Cheese_Amul                    | 30929  | 38558  | 100.74  |
| Cheese_Britannia Industries    | 21405  | 28603  | 28.92   |
| Cheese_Dynamix Dairies         | 25953  | 35962  | 32.66   |
| Cheese_Passion Cheese          | 23825  | 36961  | 58.09   |
| Curd_Amul                      | 31687  | 39436  | 30.27   |
| Curd_Mother Dairy              | 33377  | 43522  | 84.57   |
| Curd_Raj                       | 34914  | 38128  | 84.75   |
| Curd_Sudha                     | 33547  | 42341  | 76.37   |
| Ghee_Amul                      | 23120  | 30345  | 41.49   |
| Ghee_Mother Dairy              | 24667  | 35420  | 52.79   |
| Ghee_Raj                       | 25395  | 34100  | 48.13   |
| Ghee_Sudha                     | 24676  | 33007  | 95.09   |
| Ice Cream_Amul                 | 26707  | 37894  | 54.41   |
| Ice Cream_Dodla Dairy          | 26722  | 29840  | 82.24   |
| Ice Cream_Mother Dairy         | 25809  | 38762  | 94.32   |
| Ice Cream_Palle2patnam         | 24391  | 34674  | 83.73   |
| Lassi_Amul                     | 30728  | 42972  | 74.45   |
| Lassi_Mother Dairy             | 28628  | 33894  | 49.4    |
| Lassi_Raj                      | 30568  | 45762  | 93.93   |
| Lassi_Sudha                    | 23461  | 29503  | 88.05   |
| Milk_Amul                      | 21398  | 34761  | 39.24   |
| Milk_Mother Dairy              | 33619  | 40548  | 8.69    |
| Milk_Raj                       | 26355  | 43012  | 65.53   |
| Milk_Sudha                     | 23815  | 29180  | 42.34   |
| Paneer_Amul                    | 20787  | 33498  | 81.76   |
| Paneer_Mother Dairy            | 29342  | 34848  | 29.09   |
| Paneer_Raj                     | 23556  | 40347  | 87.3    |
| Paneer_Sudha                   | 28753  | 37188  | 66.7    |
| Yogurt_Amul                    | 24404  | 34347  | 89.32   |
| Yogurt_Dodla Dairy             | 26829  | 37181  | 33.81   |
| Yogurt_Mother Dairy            | 25562  | 36644  | 25.29   |
| Yogurt_Palle2patnam            | 28695  | 34303  | 84.9    |

##### Objective Function

\[
\max \sum_{i \in P} r_i x_i
\]

##### Constraints

1. **Inventory and Demand Limits**: For each product $i \in P$,
   \[
   0 \leq x_i \leq \min\{I_i, D_i\}
   \]
   (You cannot fulfill more than available inventory or more than demand.)

2. **Variable Domains**:
   \[
   x_i \geq 0 \quad \text{(continuous, for all } i \in P)
   \]

##### Complete Mathematical Model

\[
\begin{align*}
\max \quad & \sum_{i \in P} r_i x_i \\
\text{s.t.} \quad & 0 \leq x_i \leq \min\{I_i, D_i\} \quad \forall i \in P \\
& x_i \geq 0 \quad \forall i \in P
\end{align*}
\]

Where all parameter values are as listed above.