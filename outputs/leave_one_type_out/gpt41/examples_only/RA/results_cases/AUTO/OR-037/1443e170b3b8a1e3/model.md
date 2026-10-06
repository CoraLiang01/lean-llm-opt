Let the set of vehicle types be indexed by i, corresponding to the ProductName in products.csv. Let x_i be the number of vehicles of type i to order per day (x_i ≥ 0, integer).

Parameters (from products.csv and capacity.csv):

- ProductName (i): Sedan, SUV, Truck, Convertible, Minivan, Coupe, Hatchback, Station Wagon, Electric Car, Hybrid Car, Luxury Sedan, Sports Car, Crossover, Diesel Truck, Compact SUV, Luxury SUV, Cargo Van, Pickup Truck, Roadster, Muscle Car, Off-road Vehicle, Camper Van, Compact Car, Motorcycle, Electric SUV
- Value_i: profit per unit of vehicle type i (from Value column)
- Weight_i: inventory space consumed per unit of vehicle type i (from Weight column)
- Capacity: 765 (from capacity.csv)

Decision variables:
x_i = number of vehicles of type i to order per day (integer, x_i ≥ 0)

Mathematical Model:

Maximize total profit:
\[
\text{Maximize} \quad Z = 2524x_{\text{Sedan}} + 4614x_{\text{SUV}} + 8416x_{\text{Truck}} + 5917x_{\text{Convertible}} + 9048x_{\text{Minivan}} + 1140x_{\text{Coupe}} + 8962x_{\text{Hatchback}} + 1888x_{\text{Station Wagon}} + 8487x_{\text{Electric Car}} + 4425x_{\text{Hybrid Car}} + 4717x_{\text{Luxury Sedan}} + 4210x_{\text{Sports Car}} + 1226x_{\text{Crossover}} + 7400x_{\text{Diesel Truck}} + 4639x_{\text{Compact SUV}} + 7712x_{\text{Luxury SUV}} + 3299x_{\text{Cargo Van}} + 9895x_{\text{Pickup Truck}} + 4496x_{\text{Roadster}} + 4526x_{\text{Muscle Car}} + 5688x_{\text{Off-road Vehicle}} + 3007x_{\text{Camper Van}} + 3623x_{\text{Compact Car}} + 8474x_{\text{Motorcycle}} + 8372x_{\text{Electric SUV}}
\]

Subject to the inventory capacity constraint:
\[
99x_{\text{Sedan}} + 55x_{\text{SUV}} + 75x_{\text{Truck}} + 94x_{\text{Convertible}} + 80x_{\text{Minivan}} + 82x_{\text{Coupe}} + 71x_{\text{Hatchback}} + 100x_{\text{Station Wagon}} + 28x_{\text{Electric Car}} + 93x_{\text{Hybrid Car}} + 84x_{\text{Luxury Sedan}} + 83x_{\text{Sports Car}} + 62x_{\text{Crossover}} + 90x_{\text{Diesel Truck}} + 99x_{\text{Compact SUV}} + 96x_{\text{Luxury SUV}} + 21x_{\text{Cargo Van}} + 39x_{\text{Pickup Truck}} + 99x_{\text{Roadster}} + 81x_{\text{Muscle Car}} + 6x_{\text{Off-road Vehicle}} + 58x_{\text{Camper Van}} + 37x_{\text{Compact Car}} + 15x_{\text{Motorcycle}} + 37x_{\text{Electric SUV}} \leq 765
\]

Variable domains:
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \text{for all vehicle types } i
\]

Where:
- Each x_i is the number of vehicles of type i to order per day (integer, nonnegative).
- The objective maximizes total profit from all ordered vehicles.
- The constraint ensures the total inventory space used does not exceed the capacity of 765 units.