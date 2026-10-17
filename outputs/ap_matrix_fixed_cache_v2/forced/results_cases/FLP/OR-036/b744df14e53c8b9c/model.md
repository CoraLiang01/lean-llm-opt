##### Decision Variables

$x_i \in \mathbb{Z}_{\geq 0}$: Number of units of vehicle type $i$ to order daily, for each $i \in I$.

##### Parameters

Let $I$ be the set of vehicle types:
\[
I = \{
\text{Sedan},\ 
\text{SUV},\ 
\text{Truck},\ 
\text{Convertible},\ 
\text{Minivan},\ 
\text{Coupe},\ 
\text{Hatchback},\ 
\text{Station Wagon},\ 
\text{Electric Car},\ 
\text{Hybrid Car},\ 
\text{Luxury Sedan},\ 
\text{Sports Car},\ 
\text{Crossover},\ 
\text{Diesel Truck},\ 
\text{Compact SUV},\ 
\text{Luxury SUV},\ 
\text{Cargo Van},\ 
\text{Pickup Truck},\ 
\text{Roadster},\ 
\text{Muscle Car},\ 
\text{Off-road Vehicle},\ 
\text{Camper Van},\ 
\text{Compact Car},\ 
\text{Motorcycle},\ 
\text{Electric SUV}
\}
\]

Benefit coefficients $v_i$ and weights $w_i$ for each vehicle type $i$:

\[
\begin{array}{lll}
\text{Vehicle Type} & v_i\ (\text{Value}) & w_i\ (\text{Weight}) \\
\hline
\text{Sedan} & 1752 & 15 \\
\text{SUV} & 1856 & 87 \\
\text{Truck} & 8372 & 36 \\
\text{Convertible} & 6168 & 30 \\
\text{Minivan} & 9681 & 33 \\
\text{Coupe} & 8062 & 72 \\
\text{Hatchback} & 3895 & 75 \\
\text{Station Wagon} & 3254 & 71 \\
\text{Electric Car} & 1701 & 51 \\
\text{Hybrid Car} & 6799 & 21 \\
\text{Luxury Sedan} & 2724 & 97 \\
\text{Sports Car} & 6304 & 52 \\
\text{Crossover} & 3255 & 25 \\
\text{Diesel Truck} & 1923 & 15 \\
\text{Compact SUV} & 4103 & 54 \\
\text{Luxury SUV} & 4429 & 57 \\
\text{Cargo Van} & 2663 & 18 \\
\text{Pickup Truck} & 1691 & 69 \\
\text{Roadster} & 5632 & 26 \\
\text{Muscle Car} & 4793 & 38 \\
\text{Off-road Vehicle} & 1343 & 31 \\
\text{Camper Van} & 9124 & 74 \\
\text{Compact Car} & 3652 & 82 \\
\text{Motorcycle} & 8842 & 49 \\
\text{Electric SUV} & 9176 & 64 \\
\end{array}
\]

Total inventory capacity: $C = 1576$

##### Objective Function

\[
\max \sum_{i \in I} v_i x_i
\]

##### Constraints

1. Inventory capacity:
   \[
   \sum_{i \in I} w_i x_i \leq C
   \]
2. Integer and nonnegativity:
   \[
   x_i \in \mathbb{Z}_{\geq 0},\quad \forall i \in I
   \]

##### Complete Model

\[
\begin{align*}
\max\quad & 1752\,x_{\text{Sedan}} + 1856\,x_{\text{SUV}} + 8372\,x_{\text{Truck}} + 6168\,x_{\text{Convertible}} + 9681\,x_{\text{Minivan}} + 8062\,x_{\text{Coupe}} \\
& + 3895\,x_{\text{Hatchback}} + 3254\,x_{\text{Station Wagon}} + 1701\,x_{\text{Electric Car}} + 6799\,x_{\text{Hybrid Car}} + 2724\,x_{\text{Luxury Sedan}} \\
& + 6304\,x_{\text{Sports Car}} + 3255\,x_{\text{Crossover}} + 1923\,x_{\text{Diesel Truck}} + 4103\,x_{\text{Compact SUV}} + 4429\,x_{\text{Luxury SUV}} \\
& + 2663\,x_{\text{Cargo Van}} + 1691\,x_{\text{Pickup Truck}} + 5632\,x_{\text{Roadster}} + 4793\,x_{\text{Muscle Car}} + 1343\,x_{\text{Off-road Vehicle}} \\
& + 9124\,x_{\text{Camper Van}} + 3652\,x_{\text{Compact Car}} + 8842\,x_{\text{Motorcycle}} + 9176\,x_{\text{Electric SUV}} \\
\text{s.t.}\quad & 15\,x_{\text{Sedan}} + 87\,x_{\text{SUV}} + 36\,x_{\text{Truck}} + 30\,x_{\text{Convertible}} + 33\,x_{\text{Minivan}} + 72\,x_{\text{Coupe}} \\
& + 75\,x_{\text{Hatchback}} + 71\,x_{\text{Station Wagon}} + 51\,x_{\text{Electric Car}} + 21\,x_{\text{Hybrid Car}} + 97\,x_{\text{Luxury Sedan}} \\
& + 52\,x_{\text{Sports Car}} + 25\,x_{\text{Crossover}} + 15\,x_{\text{Diesel Truck}} + 54\,x_{\text{Compact SUV}} + 57\,x_{\text{Luxury SUV}} \\
& + 18\,x_{\text{Cargo Van}} + 69\,x_{\text{Pickup Truck}} + 26\,x_{\text{Roadster}} + 38\,x_{\text{Muscle Car}} + 31\,x_{\text{Off-road Vehicle}} \\
& + 74\,x_{\text{Camper Van}} + 82\,x_{\text{Compact Car}} + 49\,x_{\text{Motorcycle}} + 64\,x_{\text{Electric SUV}} \leq 1576 \\
& x_i \in \mathbb{Z}_{\geq 0},\quad \forall i \in I
\end{align*}
\]