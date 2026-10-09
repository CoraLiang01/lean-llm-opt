Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the following ProductName values:

- Sedan
- SUV
- Truck
- Convertible
- Minivan
- Coupe
- Hatchback
- Station Wagon
- Electric Car
- Hybrid Car
- Luxury Sedan
- Sports Car
- Crossover
- Diesel Truck
- Compact SUV
- Luxury SUV
- Cargo Van
- Pickup Truck
- Roadster
- Muscle Car
- Off-road Vehicle
- Camper Van
- Compact Car
- Motorcycle
- Electric SUV

Let $p_i$ be the Value (profit) for each product $i$, and $w_i$ be the Weight (space requirement) for each product $i$. The total inventory capacity is $765$.

##### Objective Function

\[
\max \sum_{i} p_i x_i
\]

where

\[
\begin{align*}
p_{\text{Sedan}} &= 2524 \\
p_{\text{SUV}} &= 4614 \\
p_{\text{Truck}} &= 8416 \\
p_{\text{Convertible}} &= 5917 \\
p_{\text{Minivan}} &= 9048 \\
p_{\text{Coupe}} &= 1140 \\
p_{\text{Hatchback}} &= 8962 \\
p_{\text{Station Wagon}} &= 1888 \\
p_{\text{Electric Car}} &= 8487 \\
p_{\text{Hybrid Car}} &= 4425 \\
p_{\text{Luxury Sedan}} &= 4717 \\
p_{\text{Sports Car}} &= 4210 \\
p_{\text{Crossover}} &= 1226 \\
p_{\text{Diesel Truck}} &= 7400 \\
p_{\text{Compact SUV}} &= 4639 \\
p_{\text{Luxury SUV}} &= 7712 \\
p_{\text{Cargo Van}} &= 3299 \\
p_{\text{Pickup Truck}} &= 9895 \\
p_{\text{Roadster}} &= 4496 \\
p_{\text{Muscle Car}} &= 4526 \\
p_{\text{Off-road Vehicle}} &= 5688 \\
p_{\text{Camper Van}} &= 3007 \\
p_{\text{Compact Car}} &= 3623 \\
p_{\text{Motorcycle}} &= 8474 \\
p_{\text{Electric SUV}} &= 8372 \\
\end{align*}
\]

##### Constraints

**Inventory Capacity Constraint:**

\[
\sum_{i} w_i x_i \leq 765
\]

where

\[
\begin{align*}
w_{\text{Sedan}} &= 99 \\
w_{\text{SUV}} &= 55 \\
w_{\text{Truck}} &= 75 \\
w_{\text{Convertible}} &= 94 \\
w_{\text{Minivan}} &= 80 \\
w_{\text{Coupe}} &= 82 \\
w_{\text{Hatchback}} &= 71 \\
w_{\text{Station Wagon}} &= 100 \\
w_{\text{Electric Car}} &= 28 \\
w_{\text{Hybrid Car}} &= 93 \\
w_{\text{Luxury Sedan}} &= 84 \\
w_{\text{Sports Car}} &= 83 \\
w_{\text{Crossover}} &= 62 \\
w_{\text{Diesel Truck}} &= 90 \\
w_{\text{Compact SUV}} &= 99 \\
w_{\text{Luxury SUV}} &= 96 \\
w_{\text{Cargo Van}} &= 21 \\
w_{\text{Pickup Truck}} &= 39 \\
w_{\text{Roadster}} &= 99 \\
w_{\text{Muscle Car}} &= 81 \\
w_{\text{Off-road Vehicle}} &= 6 \\
w_{\text{Camper Van}} &= 58 \\
w_{\text{Compact Car}} &= 37 \\
w_{\text{Motorcycle}} &= 15 \\
w_{\text{Electric SUV}} &= 37 \\
\end{align*}
\]

**Non-negativity and Integrality:**

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

##### Complete Model

\[
\begin{align*}
\max \quad & 2524\,x_{\text{Sedan}} + 4614\,x_{\text{SUV}} + 8416\,x_{\text{Truck}} + 5917\,x_{\text{Convertible}} + 9048\,x_{\text{Minivan}} + 1140\,x_{\text{Coupe}} + 8962\,x_{\text{Hatchback}} \\
& + 1888\,x_{\text{Station Wagon}} + 8487\,x_{\text{Electric Car}} + 4425\,x_{\text{Hybrid Car}} + 4717\,x_{\text{Luxury Sedan}} + 4210\,x_{\text{Sports Car}} + 1226\,x_{\text{Crossover}} \\
& + 7400\,x_{\text{Diesel Truck}} + 4639\,x_{\text{Compact SUV}} + 7712\,x_{\text{Luxury SUV}} + 3299\,x_{\text{Cargo Van}} + 9895\,x_{\text{Pickup Truck}} + 4496\,x_{\text{Roadster}} \\
& + 4526\,x_{\text{Muscle Car}} + 5688\,x_{\text{Off-road Vehicle}} + 3007\,x_{\text{Camper Van}} + 3623\,x_{\text{Compact Car}} + 8474\,x_{\text{Motorcycle}} + 8372\,x_{\text{Electric SUV}} \\
\text{s.t.} \quad & 99\,x_{\text{Sedan}} + 55\,x_{\text{SUV}} + 75\,x_{\text{Truck}} + 94\,x_{\text{Convertible}} + 80\,x_{\text{Minivan}} + 82\,x_{\text{Coupe}} + 71\,x_{\text{Hatchback}} \\
& + 100\,x_{\text{Station Wagon}} + 28\,x_{\text{Electric Car}} + 93\,x_{\text{Hybrid Car}} + 84\,x_{\text{Luxury Sedan}} + 83\,x_{\text{Sports Car}} + 62\,x_{\text{Crossover}} \\
& + 90\,x_{\text{Diesel Truck}} + 99\,x_{\text{Compact SUV}} + 96\,x_{\text{Luxury SUV}} + 21\,x_{\text{Cargo Van}} + 39\,x_{\text{Pickup Truck}} + 99\,x_{\text{Roadster}} \\
& + 81\,x_{\text{Muscle Car}} + 6\,x_{\text{Off-road Vehicle}} + 58\,x_{\text{Camper Van}} + 37\,x_{\text{Compact Car}} + 15\,x_{\text{Motorcycle}} + 37\,x_{\text{Electric SUV}} \leq 765 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\end{align*}
\]