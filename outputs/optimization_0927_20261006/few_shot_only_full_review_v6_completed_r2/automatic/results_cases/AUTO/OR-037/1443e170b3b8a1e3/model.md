**Parameters:**

Let $I$ be the set of vehicle types (from products.csv):

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

For each $i \in I$:

- $p_i$ = profit per unit (Value)
- $w_i$ = weight per unit (Weight)

**Decision Variables:**

\[
x_i = \text{number of vehicles of type } i \text{ to order per day},\quad x_i \in \mathbb{Z}_{\geq 0},\ \forall i \in I
\]

**Objective:**

\[
\max \sum_{i \in I} p_i x_i
\]

**Subject to:**

\[
\sum_{i \in I} w_i x_i \leq 765
\]
\[
x_i \in \mathbb{Z}_{\geq 0},\quad \forall i \in I
\]

**Parameter Table:**

| ProductName           | $p_i$ (Value) | $w_i$ (Weight) |
|---------------------- |:-------------:|:--------------:|
| Sedan                 | 2524          | 99             |
| SUV                   | 4614          | 55             |
| Truck                 | 8416          | 75             |
| Convertible           | 5917          | 94             |
| Minivan               | 9048          | 80             |
| Coupe                 | 1140          | 82             |
| Hatchback             | 8962          | 71             |
| Station Wagon         | 1888          | 100            |
| Electric Car          | 8487          | 28             |
| Hybrid Car            | 4425          | 93             |
| Luxury Sedan          | 4717          | 84             |
| Sports Car            | 4210          | 83             |
| Crossover             | 1226          | 62             |
| Diesel Truck          | 7400          | 90             |
| Compact SUV           | 4639          | 99             |
| Luxury SUV            | 7712          | 96             |
| Cargo Van             | 3299          | 21             |
| Pickup Truck          | 9895          | 39             |
| Roadster              | 4496          | 99             |
| Muscle Car            | 4526          | 81             |
| Off-road Vehicle      | 5688          | 6              |
| Camper Van            | 3007          | 58             |
| Compact Car           | 3623          | 37             |
| Motorcycle            | 8474          | 15             |
| Electric SUV          | 8372          | 37             |

**Capacity:**

\[
\text{Total inventory capacity: } 765
\]

**Complete Model:**

\[
\begin{align*}
\max\quad & 2524\,x_{\text{Sedan}} + 4614\,x_{\text{SUV}} + 8416\,x_{\text{Truck}} + 5917\,x_{\text{Convertible}} + 9048\,x_{\text{Minivan}} + 1140\,x_{\text{Coupe}} + 8962\,x_{\text{Hatchback}} \\
& + 1888\,x_{\text{Station Wagon}} + 8487\,x_{\text{Electric Car}} + 4425\,x_{\text{Hybrid Car}} + 4717\,x_{\text{Luxury Sedan}} + 4210\,x_{\text{Sports Car}} + 1226\,x_{\text{Crossover}} \\
& + 7400\,x_{\text{Diesel Truck}} + 4639\,x_{\text{Compact SUV}} + 7712\,x_{\text{Luxury SUV}} + 3299\,x_{\text{Cargo Van}} + 9895\,x_{\text{Pickup Truck}} + 4496\,x_{\text{Roadster}} \\
& + 4526\,x_{\text{Muscle Car}} + 5688\,x_{\text{Off-road Vehicle}} + 3007\,x_{\text{Camper Van}} + 3623\,x_{\text{Compact Car}} + 8474\,x_{\text{Motorcycle}} + 8372\,x_{\text{Electric SUV}} \\
\text{s.t.}\quad & 99\,x_{\text{Sedan}} + 55\,x_{\text{SUV}} + 75\,x_{\text{Truck}} + 94\,x_{\text{Convertible}} + 80\,x_{\text{Minivan}} + 82\,x_{\text{Coupe}} + 71\,x_{\text{Hatchback}} \\
& + 100\,x_{\text{Station Wagon}} + 28\,x_{\text{Electric Car}} + 93\,x_{\text{Hybrid Car}} + 84\,x_{\text{Luxury Sedan}} + 83\,x_{\text{Sports Car}} + 62\,x_{\text{Crossover}} \\
& + 90\,x_{\text{Diesel Truck}} + 99\,x_{\text{Compact SUV}} + 96\,x_{\text{Luxury SUV}} + 21\,x_{\text{Cargo Van}} + 39\,x_{\text{Pickup Truck}} + 99\,x_{\text{Roadster}} \\
& + 81\,x_{\text{Muscle Car}} + 6\,x_{\text{Off-road Vehicle}} + 58\,x_{\text{Camper Van}} + 37\,x_{\text{Compact Car}} + 15\,x_{\text{Motorcycle}} + 37\,x_{\text{Electric SUV}} \leq 765 \\
& x_i \in \mathbb{Z}_{\geq 0},\quad \forall i \in I
\end{align*}
\]