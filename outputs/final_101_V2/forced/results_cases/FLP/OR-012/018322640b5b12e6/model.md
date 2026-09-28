##### Decision Variables

Let $x_i \geq 0$ be the number of units of product $i$ to fulfill for customer purchases, for each product $i$ in the set of all products $P$.

##### Parameters

For each product $i \in P$:

- $r_i$: revenue per unit of product $i$ (from the 'Revenue' column)
- $s_i$: initial inventory of product $i$ (from the 'Initial Inventory' column)
- $d_i$: demand for product $i$ (from the 'Demand' column)

The set of products $P$ and their parameters are:

| Product Name                                         | $r_i$   | $s_i$ | $d_i$ |
|------------------------------------------------------|---------|-------|-------|
| 1984 by George Orwell                                | 19.99   | 20    | 6     |
| Adidas 3-Stripes Shorts                              | 29.99   | 30    | 7     |
| Adidas Essential Track Pants                         | 44.99   | 20    | 5     |
| Adidas FIFA World Cup Football                       | 29.99   | 20    | 4     |
| Adidas Originals Superstar Sneakers                  | 79.99   | 20    | 6     |
| Adidas Originals Trefoil Hoodie                      | 64.99   | 20    | 6     |
| Adidas Ultraboost Running Shoes                      | 179.99  | 10    | 3     |
| Adidas Ultraboost Shoes                              | 179.99  | 10    | 3     |
| Amazon Echo Dot (4th Gen)                            | 49.99   | 20    | 5     |
| Amazon Echo Show 10                                  | 249.99  | 10    | 2     |
| Amazon Fire TV Stick 4K                              | 49.99   | 20    | 5     |
| Anastasia Beverly Hills Brow Wiz                     | 23      | 10    | 3     |
| Anker PowerCore Portable Charger                     | 59.99   | 20    | 6     |
| Anova Precision Cooker                               | 199     | 10    | 3     |
| Anova Precision Oven                                 | 599     | 10    | 2     |
| Apple AirPods Max                                    | 549     | 10    | 2     |
| Apple AirPods Pro                                    | 249.99  | 10    | 3     |
| Apple MacBook Air                                    | 1199.99 | 10    | 2     |
| Apple MacBook Pro 16-inch                            | 2399    | 10    | 2     |
| Apple TV 4K                                          | 179     | 10    | 3     |
| Apple Watch Series 8                                 | 399.99  | 20    | 4     |
| Apple iPad Air                                       | 599.99  | 10    | 3     |
| Atomic Habits by James Clear                         | 16.99   | 20    | 6     |
| Babolat Pure Drive Tennis Racket                     | 199.99  | 20    | 5     |
| Becoming by Michelle Obama                           | 32.5    | 20    | 6     |
| Biore UV Aqua Rich Watery Essence Sunscreen          | 15      | 10    | 2     |
| Blueair Classic 480i                                 | 599.99  | 10    | 3     |
| Bose QuietComfort 35 Headphones                      | 299.99  | 10    | 2     |
| Bose QuietComfort 35 II Wireless Headphones          | 299     | 10    | 2     |
| Bose SoundLink Color Bluetooth Speaker II            | 129     | 10    | 2     |
| Bose SoundLink Revolve+ Speaker                      | 299.99  | 20    | 5     |
| Bose SoundSport Wireless Earbuds                     | 149.99  | 10    | 3     |
| Bowflex SelectTech 1090 Adjustable Dumbbells         | 699.99  | 10    | 2     |
| Bowflex SelectTech 552 Dumbbells                     | 399.99  | 10    | 2     |
| Breville Nespresso Creatista Plus                    | 499.95  | 10    | 2     |
| Breville Smart Coffee Grinder Pro                    | 199.95  | 10    | 2     |
| Breville Smart Grill                                 | 299.95  | 10    | 3     |
| Breville Smart Oven                                  | 299.99  | 10    | 2     |
| Calvin Klein Boxer Briefs                            | 29.99   | 30    | 7     |
| Canon EOS R5 Camera                                 | 3899.99 | 10    | 2     |
| Canon EOS Rebel T7i DSLR Camera                      | 749.99  | 10    | 2     |
| Caudalie Vinoperfect Radiance Serum                  | 79      | 10    | 2     |
| CeraVe Hydrating Facial Cleanser                     | 14.99   | 10    | 3     |
| Champion Reverse Weave Hoodie                        | 49.99   | 20    | 5     |
| Chanel No. 5 Perfume                                 | 129.99  | 10    | 2     |
| Charlotte Tilbury Magic Cream                        | 100     | 10    | 2     |
| Clinique Dramatically Different Moisturizing Lotion  | 29.5    | 10    | 2     |
| Clinique Moisture Surge                             | 52      | 10    | 2     |
| Columbia Fleece Jacket                               | 59.99   | 20    | 6     |
| Crock-Pot 6-Quart Slow Cooker                        | 49.99   | 10    | 3     |
| Cuisinart Coffee Center                              | 199.95  | 10    | 3     |
| Cuisinart Custom 14-Cup Food Processor               | 199.99  | 10    | 2     |
| Cuisinart Griddler Deluxe                            | 159.99  | 10    | 2     |
| De'Longhi Magnifica Espresso Machine                 | 899.99  | 10    | 2     |
| Dr. Jart+ Cicapair Tiger Grass Color Correcting Treatment | 52 | 10    | 2     |
| Drunk Elephant C-Firma Day Serum                     | 78      | 10    | 2     |
| Dune by Frank Herbert                                | 25.99   | 20    | 6     |
| Dyson Pure Cool Link                                 | 499.99  | 10    | 2     |
| Dyson Supersonic Hair Dryer                          | 399.99  | 20    | 5     |
| Dyson V11 Vacuum                                     | 499.99  | 10    | 2     |
| Dyson V8 Absolute                                    | 399.99  | 10    | 2     |
| Educated by Tara Westover                            | 28      | 20    | 4     |
| Estee Lauder Advanced Night Repair                   | 105     | 10    | 2     |
| Eufy RoboVac 11S                                     | 219.99  | 20    | 5     |
| Fenty Beauty Killawatt Highlighter                   | 36      | 10    | 2     |
| First Aid Beauty Ultra Repair Cream                  | 34      | 10    | 3     |
| Fitbit Charge 5                                      | 129.99  | 10    | 3     |
| Fitbit Inspire 2                                     | 99.95   | 10    | 3     |
| Fitbit Luxe                                          | 149.95  | 10    | 3     |
| Fitbit Versa 3                                       | 229.95  | 20    | 5     |
| Forever 21 Graphic Tee                               | 12.99   | 30    | 7     |
| Fresh Sugar Lip Treatment                            | 24      | 10    | 2     |
| Gap 1969 Original Fit Jeans                          | 59.99   | 20    | 5     |
| Gap Crewneck Sweatshirt                              | 34.99   | 20    | 6     |
| Gap Essential Crewneck T-Shirt                       | 19.99   | 30    | 8     |
| Gap High Rise Skinny Jeans                           | 49.99   | 20    | 5     |
| Garmin Edge 530                                      | 299.99  | 10    | 3     |
| Garmin Fenix 6X Pro                                  | 999.99  | 10    | 2     |
| Garmin Forerunner 245                                | 299.99  | 10    | 2     |
| Garmin Forerunner 945                                | 499.99  | 20    | 5     |
| GlamGlow Supermud Clearing Treatment                 | 59      | 10    | 2     |
| Glossier Boy Brow                                    | 16      | 10    | 3     |
| Glossier Cloud Paint                                 | 18      | 10    | 2     |
| GoPro HERO10 Black                                   | 399.99  | 20    | 4     |
| GoPro HERO9 Black                                    | 449.99  | 10    | 2     |
| Gone Girl by Gillian Flynn                           | 22.99   | 10    | 3     |
| Google Nest Hub Max                                  | 229.99  | 10    | 3     |
| Google Nest Wifi Router                              | 169     | 10    | 2     |
| Google Pixel 6 Pro                                   | 899.99  | 10    | 2     |
| Google Pixelbook Go                                  | 649.99  | 10    | 2     |
| H&M Slim Fit Jeans                                   | 39.99   | 20    | 5     |
| HP Spectre x360 Laptop                               | 1599.99 | 10    | 2     |
| Hamilton Beach FlexBrew Coffee Maker                 | 89.99   | 10    | 2     |
| Hanes ComfortSoft T-Shirt                            | 9.99    | 50    | 15    |
| Harry Potter and the Sorcerer's Stone                | 24.99   | 20    | 4     |
| Hydro Flask Standard Mouth Water Bottle              | 32.95   | 20    | 5     |
| Hydro Flask Wide Mouth Water Bottle                  | 39.95   | 20    | 6     |
| Hyperice Hypervolt Massager                          | 349     | 10    | 2     |
| Instant Pot Duo                                      | 89.99   | 20    | 4     |
| Instant Pot Duo Crisp                                | 179.99  | 10    | 2     |
| Instant Pot Duo Evo Plus                             | 139.99  | 10    | 3     |
| Instant Pot Duo Nova                                 | 99.95   | 10    | 2     |
| Instant Pot Ultra                                    | 139.99  | 10    | 2     |
| Keurig K-Elite Coffee Maker                          | 189.99  | 20    | 5     |
| Keurig K-Mini Coffee Maker                           | 79.99   | 10    | 3     |
| Kiehl's Midnight Recovery Concentrate                | 82      | 10    | 2     |
| Kindle Paperwhite                                    | 129.99  | 10    | 3     |
| KitchenAid Artisan Stand Mixer                       | 499.99  | 10    | 2     |
| KitchenAid Stand Mixer                               | 379.99  | 10    | 2     |
| L'Occitane Shea Butter Hand Cream                    | 29      | 10    | 3     |
| L'Oreal Revitalift Serum                             | 39.99   | 10    | 3     |
| LG OLED TV                                           | 1299.99 | 10    | 3     |
| La Mer Cr_¨me de la Mer Moisturizer                  | 190     | 10    | 2     |
| Lancome La Vie Est Belle                             | 102     | 10    | 2     |
| Laneige Water Sleeping Mask                          | 25      | 10    | 2     |
| Levi's 501 Jeans                                     | 69.99   | 20    | 5     |
| Levi's 511 Slim Fit Jeans                            | 59.99   | 20    | 5     |
| Levi's Sherpa Trucker Jacket                         | 98      | 10    | 3     |
| Levi's Trucker Jacket                                | 89.99   | 10    | 3     |

##### Objective Function

\[
\max \sum_{i \in P} r_i x_i
\]

##### Constraints

1. Inventory and demand fulfillment for each product:
   \[
   0 \leq x_i \leq \min\{s_i, d_i\}, \quad \forall i \in P
   \]

##### Variable Domains

\[
x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in P
\]
(If partial units are not allowed; otherwise, $x_i$ can be continuous.)

##### Summary

- $P$ is the set of all products listed above.
- $r_i$ is the revenue per unit for product $i$.
- $s_i$ is the initial inventory for product $i$.
- $d_i$ is the demand for product $i$.
- $x_i$ is the number of units of product $i$ to fulfill, subject to $0 \leq x_i \leq \min\{s_i, d_i\}$.

The model maximizes total revenue by optimally allocating available inventory to meet as much demand as possible for each product, without exceeding either the available inventory or the demand for any product.