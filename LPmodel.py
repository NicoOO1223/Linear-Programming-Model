import pandas as pd
import pulp

######################   Data, Vectors and Matrices   #########################
# Read the data from the Excel file
data = pd.read_excel('Data.xlsx', sheet_name=None)

# Grouping locations by 'LocationTYpe'
locationDict = data['Nodes'].groupby('LocationTYpe')['Location'].apply(list).to_dict()

# Extracting locations for different types
Beneficiary_camps = locationDict.get('Beneficiary Camp', [])
Ports = locationDict.get('Port', [])
Suppliers = locationDict.get('Supplier', [])
Warehouses = locationDict.get('Warehouse', [])

# Extract commodities and nutrients lists
commodities = data['Commodities'].iloc[:, 0].tolist()
nutrients = data['Nutrients'].iloc[:, 0].tolist()

nutrient_data = data['Nutritional values']
nutrient_content = nutrient_data.set_index('Commodity (100g)').to_dict(orient='index')

req_nutrients = data['Total nutrients per camp']
req_nutrients = req_nutrients.drop(columns=['Beneficiaries'])
nutrient_req = req_nutrients.set_index('Location').to_dict(orient='index')


# Creating a dictionary for procurement prices, with commodities as keys and suppliers as subkeys
cij = data['Procurement'].groupby(['Commodity', 'Supplier'])['Procurement price ($/ton)'].first().unstack(fill_value=0).to_dict()

# Create the dictionary for shipping costs (cijk)
cijk = {}

for _, row in data['SeaTransport'].iterrows():
    origin = row['Origin']
    destination = row['Destination']
    commodity = row['Commodity']
    cost = row['SeaTransport cost ($/ton)']
    
    # Initialize the nested dictionary if not already present
    if origin not in cijk:
        cijk[origin] = {}
    if commodity not in cijk[origin]:
        cijk[origin][commodity] = {}
    
    cijk[origin][commodity][destination] = cost


# Create a dictionary with Port names as keys and their corresponding handling costs as values
port_data = data['Nodes'][data['Nodes']['LocationTYpe'] == 'Port'][['Location', 'Port capacity (mt/month)', 'Handling cost ($/ton)']]

h_p = port_data.set_index('Location')['Handling cost ($/ton)'].to_dict() #Handling cost at ports
Cap_p = port_data.set_index('Location')['Port capacity (mt/month)'].to_dict() #Ports capacity


port_to_warehouse = data['LandTransport'].iloc[:11]  
warehouse_to_village = data['LandTransport'].iloc[11:]  


c_lk = {}

for _, row in port_to_warehouse.iterrows():
    origin = row['Origin']
    destination = row['Destination']
    cost = row['Landtransport cost ($/ton)']
    
    # Initialize the nested dictionary if not already present
    if origin not in c_lk:
        c_lk[origin] = {}
    
    c_lk[origin][destination] = cost

# Extracting warehouse handling costs
warehouse_data = data['Nodes'][data['Nodes']['LocationTYpe'] == 'Warehouse'][['Location', 'Port capacity (mt/month)', 'Handling cost ($/ton)']]

# Create a dictionary with Warehouse names as keys and their corresponding handling costs as values
h_w = warehouse_data.set_index('Location')['Handling cost ($/ton)'].to_dict()
Cap_w = warehouse_data.set_index('Location')['Port capacity (mt/month)'].to_dict()

c_km = {}

for _, row in warehouse_to_village.iterrows():
    origin = row['Origin']
    destination = row['Destination']
    cost = row['Landtransport cost ($/ton)']
    
    # Initialize the nested dictionary if not already present
    if origin not in c_km:
        c_km[origin] = {}
    
    c_km[origin][destination] = cost


### MODEL ###
lp = pulp.LpProblem('Optimizating', pulp.LpMinimize)

x_ij = pulp.LpVariable.dicts('purchase', (commodities, Suppliers), lowBound=0, cat='Continuous')
y_ijk = pulp.LpVariable.dicts('ship', (commodities, Suppliers, Ports), lowBound=0, cat='Continuous')
z_ikl = pulp.LpVariable.dicts('land', (commodities, Ports, Warehouses), lowBound=0, cat='Continuous')
w_ilm = pulp.LpVariable.dicts('land', (commodities, Warehouses, Beneficiary_camps), lowBound=0, cat='Continuous')

#Objective function
lp += pulp.lpSum(cij[supplier][commodity] * x_ij[commodity][supplier] 
                 for supplier in Suppliers 
                 for commodity in commodities) \
     + pulp.lpSum(cijk.get(supplier, {}).get(commodity, {}).get(port, 0) * y_ijk[commodity][supplier][port] 
                 for supplier in Suppliers 
                 for commodity in commodities 
                 for port in Ports) \
     + pulp.lpSum(h_p[port] * pulp.lpSum(y_ijk[commodity][supplier][port] 
                    for commodity in commodities 
                    for supplier in Suppliers) 
                for port in Ports) \
     + pulp.lpSum(c_lk.get(port, {}).get(warehouse, 0) * z_ikl[commodity][port][warehouse]
                 for commodity in commodities
                 for port in Ports
                 for warehouse in Warehouses) \
    + pulp.lpSum(h_w[warehouse] * pulp.lpSum(z_ikl[commodity][port][warehouse] 
                    for port in Ports 
                    for commodity in commodities) 
                for warehouse in Warehouses) \
    + pulp.lpSum(c_km.get(warehouse, {}).get(village, 0) * w_ilm[commodity][warehouse][village]
                 for commodity in commodities 
                 for warehouse in Warehouses
                 for village in Beneficiary_camps)

# Port Capacity Constraint
for k in Ports:
    lp += pulp.lpSum(y_ijk[i][j][k] for i in commodities for j in Suppliers) <= Cap_p.get(k, 0), f"PortCapacity_{k}"

# Adding warehouse capacity constraints
for l in Warehouses:
    lp += pulp.lpSum(z_ikl[i][k][l] for i in commodities for k in Ports) <= Cap_w.get(l, 0), f"WarehouseCapacity_{l}"

### Balance Constraints
for j in Suppliers:
    for i in commodities:
        lp += pulp.lpSum(y_ijk[i][j][k] for k in Ports) == x_ij[i][j], f"SupplierBalance_{i}_{j}"

for k in Ports:
    for i in commodities:
        lp += pulp.lpSum(y_ijk[i][j][k] for j in Suppliers) == pulp.lpSum(z_ikl[i][k][l] for l in Warehouses), f"PortBalance_{i}_{k}"

for l in Warehouses:
    for i in commodities:
        lp += pulp.lpSum(z_ikl[i][k][l] for k in Ports) == pulp.lpSum(w_ilm[i][l][m] for m in Beneficiary_camps), f"WarehouseBalance_{i}_{l}"

#Nutritional Constraint
for m in nutrient_req:  # Loop through each village
    for n in nutrient_req[m]:  # Loop through each nutrient in the village
        lp += pulp.lpSum(
            w_ilm[i][l][m] * nutrient_content[i][n]  # Amount of nutrient n from commodity i shipped from warehouse l to village m
            for i in commodities  # For each commodity
            for l in Warehouses  # For each warehouse
        ) >= nutrient_req[m][n], f"NutrientRequirement_{m}_{n}"

        
lp.solve(pulp.PULP_CBC_CMD(msg=True))
# Check if the solver found an optimal solution
if pulp.LpStatus[lp.status] == 'Optimal':
    print("Optimal solution found!")
else:
    print(f"Solution status: {pulp.LpStatus[lp.status]}")