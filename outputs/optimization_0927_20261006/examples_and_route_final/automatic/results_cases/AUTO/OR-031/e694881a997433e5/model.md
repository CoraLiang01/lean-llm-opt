Let $i$ index the dairy products, with identifiers as in the Full_Product_Name column. Let $x_i$ be the number of units of product $i$ to be fulfilled.

Parameters:
- $r_i$: Revenue per unit of product $i$ (from Revenue column)
- $d_i$: Demand for product $i$ (from Demand column)
- $s_i$: Initial Inventory for product $i$ (from Initial Inventory column)

Decision variables:
- $x_i \in \mathbb{Z}_{\geq 0}$, for all $i$ (number of units of product $i$ fulfilled)

Objective:
\[
\max \sum_{i} r_i x_i
\]

Subject to, for all products $i$:
\[
0 \leq x_i \leq \min\{d_i,\, s_i\}
\]
\[
x_i \in \mathbb{Z}_{\geq 0}
\]

Explicitly, for each product (in source order):

\[
\begin{align*}
&\max \Bigg[ 
96.86\, x_{\text{Butter\_Amul}} + 
48.01\, x_{\text{Butter\_Mother Dairy}} + 
8.83\, x_{\text{Butter\_Parag Milk Foods}} + 
92.96\, x_{\text{Butter\_Warana}} + \\
&\qquad 40.75\, x_{\text{Buttermilk\_Amul}} + 
83.07\, x_{\text{Buttermilk\_Mother Dairy}} + 
15.64\, x_{\text{Buttermilk\_Raj}} + 
56.57\, x_{\text{Buttermilk\_Sudha}} + \\
&\qquad 100.74\, x_{\text{Cheese\_Amul}} + 
28.92\, x_{\text{Cheese\_Britannia Industries}} + 
32.66\, x_{\text{Cheese\_Dynamix Dairies}} + 
58.09\, x_{\text{Cheese\_Passion Cheese}} + \\
&\qquad 30.27\, x_{\text{Curd\_Amul}} + 
84.57\, x_{\text{Curd\_Mother Dairy}} + 
84.75\, x_{\text{Curd\_Raj}} + 
76.37\, x_{\text{Curd\_Sudha}} + \\
&\qquad 41.49\, x_{\text{Ghee\_Amul}} + 
52.79\, x_{\text{Ghee\_Mother Dairy}} + 
48.13\, x_{\text{Ghee\_Raj}} + 
95.09\, x_{\text{Ghee\_Sudha}} + \\
&\qquad 54.41\, x_{\text{Ice Cream\_Amul}} + 
82.24\, x_{\text{Ice Cream\_Dodla Dairy}} + 
94.32\, x_{\text{Ice Cream\_Mother Dairy}} + 
83.73\, x_{\text{Ice Cream\_Palle2patnam}} + \\
&\qquad 74.45\, x_{\text{Lassi\_Amul}} + 
49.4\, x_{\text{Lassi\_Mother Dairy}} + 
93.93\, x_{\text{Lassi\_Raj}} + 
88.05\, x_{\text{Lassi\_Sudha}} + \\
&\qquad 39.24\, x_{\text{Milk\_Amul}} + 
8.69\, x_{\text{Milk\_Mother Dairy}} + 
65.53\, x_{\text{Milk\_Raj}} + 
42.34\, x_{\text{Milk\_Sudha}} + \\
&\qquad 81.76\, x_{\text{Paneer\_Amul}} + 
29.09\, x_{\text{Paneer\_Mother Dairy}} + 
87.3\, x_{\text{Paneer\_Raj}} + 
66.7\, x_{\text{Paneer\_Sudha}} + \\
&\qquad 89.32\, x_{\text{Yogurt\_Amul}} + 
33.81\, x_{\text{Yogurt\_Dodla Dairy}} + 
25.29\, x_{\text{Yogurt\_Mother Dairy}} + 
84.9\, x_{\text{Yogurt\_Palle2patnam}}
\Bigg]
\end{align*}
\]

Subject to, for each product $i$ (using the Full_Product_Name as $i$):

\[
0 \leq x_i \leq \min\{\text{Demand}_i,\, \text{Initial Inventory}_i\}
\]
\[
x_i \in \mathbb{Z}_{\geq 0}
\]

Where the parameters for each product are:

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

That is, for each $i$:
\[
0 \leq x_i \leq \min\{\text{Demand}_i,\, \text{Initial Inventory}_i\},\quad x_i \in \mathbb{Z}_{\geq 0}
\]