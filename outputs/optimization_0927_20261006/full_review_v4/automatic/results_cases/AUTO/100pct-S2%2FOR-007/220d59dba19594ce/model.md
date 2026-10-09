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

Let $p_i$ be the Value (profit) for product $i$, and $w_i$ be the Weight for product $i$ (inventory space required per unit). The total inventory capacity is $765$.

##### Objective Function

\[
\max \quad 2524\,x_{\text{Sedan}} + 4614\,x_{\text{SUV}} + 8416\,x_{\text{Truck}} + 5917\,x_{\text{Convertible}} + 9048\,x_{\text{Minivan}} + 1140\,x_{\text{Coupe}} + 8962\,x_{\text{Hatchback}} + 1888\,x_{\text{Station Wagon}} + 8487\,x_{\text{Electric Car}} + 4425\,x_{\text{Hybrid Car}} + 4717\,x_{\text{Luxury Sedan}} + 4210\,x_{\text{Sports Car}} + 1226\,x_{\text{Crossover}} + 7400\,x_{\text{Diesel Truck}} + 4639\,x_{\text{Compact SUV}} + 7712\,x_{\text{Luxury SUV}} + 3299\,x_{\text{Cargo Van}} + 9895\,x_{\text{Pickup Truck}} + 4496\,x_{\text{Roadster}} + 4526\,x_{\text{Muscle Car}} + 5688\,x_{\text{Off-road Vehicle}} + 3007\,x_{\text{Camper Van}} + 3623\,x_{\text{Compact Car}} + 8474\,x_{\text{Motorcycle}} + 8372\,x_{\text{Electric SUV}}
\]

##### Subject to

\[
99\,x_{\text{Sedan}} + 55\,x_{\text{SUV}} + 75\,x_{\text{Truck}} + 94\,x_{\text{Convertible}} + 80\,x_{\text{Minivan}} + 82\,x_{\text{Coupe}} + 71\,x_{\text{Hatchback}} + 100\,x_{\text{Station Wagon}} + 28\,x_{\text{Electric Car}} + 93\,x_{\text{Hybrid Car}} + 84\,x_{\text{Luxury Sedan}} + 83\,x_{\text{Sports Car}} + 62\,x_{\text{Crossover}} + 90\,x_{\text{Diesel Truck}} + 99\,x_{\text{Compact SUV}} + 96\,x_{\text{Luxury SUV}} + 21\,x_{\text{Cargo Van}} + 39\,x_{\text{Pickup Truck}} + 99\,x_{\text{Roadster}} + 81\,x_{\text{Muscle Car}} + 6\,x_{\text{Off-road Vehicle}} + 58\,x_{\text{Camper Van}} + 37\,x_{\text{Compact Car}} + 15\,x_{\text{Motorcycle}} + 37\,x_{\text{Electric SUV}} \leq 765
\]

\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

##### Where

- $x_i$ = number of vehicles of type $i$ to order per day (nonnegative integer)
- $p_i$ = Value (profit) for product $i$ (see above)
- $w_i$ = Weight for product $i$ (see above)
- Total inventory capacity = $765$

All coefficients and identifiers are as retrieved from the data.