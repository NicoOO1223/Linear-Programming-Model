import pandas as pd
import pulp

######################   Data, Vectors and Matrices   #########################

# Read all sheets from the Excel file into a dictionary of DataFrames
data = pd.read_excel('Data.xlsx', sheet_name=None)

# Group locations by their type (e.g., Port, Supplier, Warehouse, Beneficiary Camp)
locationDict = data['Nodes'].groupby('LocationTYpe')['Location'].apply(list).to_dict()

# Extract specific lists of locations by type
Beneficiary_camps = locationDict.get('Beneficiary Camp', [])
Ports = locationDict.get('Port', [])
Suppliers = locationDict.get('Supplier', [])
Warehouses = locationDict.get('Warehouse', [])

# Get list of commodities and nutrients
commodities = data['Commodities'].iloc[:, 0].tolist()
nutrients = data['Nutrients'].iloc[:, 0].tolist()

# Nutrient content per 100g, scaled to match units (multiplied by 10000 to represent per ton)
nutrient_data = data['Nutritional values']
nutrient_data_scaled = nutrient_data.copy()
nutrient_data_scaled.update(nutrient_data_scaled.drop(columns=['Commodity (100g)']) * 10000)

# Convert the nutrient content to a nested dictionary: {commodity: {nutrient: value}}
nutrient_content = nutrient_data_scaled.set_index('Commodity (100g)').to_dict(orient='index')

# Required nutrient amounts at each camp (village), ignore 'Beneficiaries' column
req_nutrients = data['Total nutrients per camp'].drop(columns=['Beneficiaries'])
nutrient_req = req_nutrients.set_index('Location').to_dict(orient='index')


# Procurement cost: {supplier: {commodity: cost}}
cij = data['Procurement'].groupby(['Commodity', 'Supplier'])['Procurement price ($/ton)'].first().unstack(fill_value=1000000).to_dict()

# Sea transport cost: {supplier: {commodity: {port: cost}}}
cijk = {}
for _, row in data['SeaTransport'].iterrows():
    origin, destination, commodity, cost = row['Origin'], row['Destination'], row['Commodity'], row['SeaTransport cost ($/ton)']
    cijk.setdefault(origin, {}).setdefault(commodity, {})[destination] = cost

# Port handling cost and capacity: {port: cost/capacity}
port_data = data['Nodes'][data['Nodes']['LocationTYpe'] == 'Port'][['Location', 'Port capacity (mt/month)', 'Handling cost ($/ton)']]
h_p = port_data.set_index('Location')['Handling cost ($/ton)'].to_dict()
Cap_p = port_data.set_index('Location')['Port capacity (mt/month)'].to_dict()

# Split land transport into port-to-warehouse and warehouse-to-village
port_to_warehouse = data['LandTransport'].iloc[:11]  
warehouse_to_village = data['LandTransport'].iloc[11:]  

# Land transport cost (port to warehouse): {port: {warehouse: cost}}
c_lk = {}
for _, row in port_to_warehouse.iterrows():
    origin, destination, cost = row['Origin'], row['Destination'], row['Landtransport cost ($/ton)']
    c_lk.setdefault(origin, {})[destination] = cost

# Warehouse handling cost and capacity: {warehouse: cost/capacity}
warehouse_data = data['Nodes'][data['Nodes']['LocationTYpe'] == 'Warehouse'][['Location', 'Port capacity (mt/month)', 'Handling cost ($/ton)']]
h_w = warehouse_data.set_index('Location')['Handling cost ($/ton)'].to_dict()
Cap_w = warehouse_data.set_index('Location')['Port capacity (mt/month)'].to_dict()

# Land transport cost (warehouse to village): {warehouse: {village: cost}}
c_km = {}
for _, row in warehouse_to_village.iterrows():
    origin, destination, cost = row['Origin'], row['Destination'], row['Landtransport cost ($/ton)']
    c_km.setdefault(origin, {})[destination] = cost


########################### Linear Programming Model ###########################

# Create a minimization problem
lp = pulp.LpProblem('Optimizating', pulp.LpMinimize)

# Define decision variables for each stage of the supply chain
x_ij = pulp.LpVariable.dicts('purchase', (commodities, Suppliers), lowBound=0, cat='Continuous')  # Supplier to initial procurement
y_ijk = pulp.LpVariable.dicts('ship', (commodities, Suppliers, Ports), lowBound=0, cat='Continuous')  # Supplier to port
z_ikl = pulp.LpVariable.dicts('land', (commodities, Ports, Warehouses), lowBound=0, cat='Continuous')  # Port to warehouse
w_ilm = pulp.LpVariable.dicts('land', (commodities, Warehouses, Beneficiary_camps), lowBound=0, cat='Continuous')  # Warehouse to village

########################### Objective Function ##################################

# Total cost = Procurement + Sea transport + Port handling + Port-Warehouse land transport
# + Warehouse handling + Warehouse-Village land transport
lp += (
    pulp.lpSum(cij[supplier][commodity] * x_ij[commodity][supplier]
               for supplier in Suppliers for commodity in commodities if cij[supplier][commodity] > 0) +

    pulp.lpSum(cijk.get(supplier, {}).get(commodity, {}).get(port, 10000000) * y_ijk[commodity][supplier][port]
               for supplier in Suppliers for commodity in commodities for port in Ports) +

    pulp.lpSum(h_p[port] * pulp.lpSum(y_ijk[commodity][supplier][port]
               for commodity in commodities for supplier in Suppliers)
               for port in Ports) +

    pulp.lpSum(c_lk.get(port, {}).get(warehouse, 10000000) * z_ikl[commodity][port][warehouse]
               for commodity in commodities for port in Ports for warehouse in Warehouses) +

    pulp.lpSum(h_w[warehouse] * pulp.lpSum(z_ikl[commodity][port][warehouse]
               for port in Ports for commodity in commodities)
               for warehouse in Warehouses) +

    pulp.lpSum(c_km.get(warehouse, {}).get(village, 10000000) * w_ilm[commodity][warehouse][village]
               for commodity in commodities for warehouse in Warehouses for village in Beneficiary_camps)
)

########################### Constraints ########################################

# Port Capacity: Total incoming shipment to a port can't exceed its capacity
for k in Ports:
    lp += pulp.lpSum(y_ijk[i][j][k] for i in commodities for j in Suppliers) <= Cap_p.get(k, 0), f"PortCapacity_{k}"

# Warehouse Capacity: Total incoming shipment to a warehouse can't exceed its capacity
for l in Warehouses:
    lp += pulp.lpSum(z_ikl[i][k][l] for i in commodities for k in Ports) <= Cap_w.get(l, 0), f"WarehouseCapacity_{l}"

# Supplier Balance: All products sent from a supplier to ports must match purchased quantity
for j in Suppliers:
    for i in commodities:
        lp += pulp.lpSum(y_ijk[i][j][k] for k in Ports) == x_ij[i][j], f"SupplierBalance_{i}_{j}"

# Port Balance: All items entering a port must leave to some warehouse
for k in Ports:
    for i in commodities:
        lp += pulp.lpSum(y_ijk[i][j][k] for j in Suppliers) == pulp.lpSum(z_ikl[i][k][l] for l in Warehouses), f"PortBalance_{i}_{k}"

# Warehouse Balance: All items entering a warehouse must be sent to some village
for l in Warehouses:
    for i in commodities:
        lp += pulp.lpSum(z_ikl[i][k][l] for k in Ports) == pulp.lpSum(w_ilm[i][l][m] for m in Beneficiary_camps), f"WarehouseBalance_{i}_{l}"

# Nutritional Constraints: Ensure each village receives enough nutrients of each type
for m in nutrient_req:
    for n in nutrient_req[m]:
        lp += pulp.lpSum(
            w_ilm[i][l][m] * nutrient_content[i][n]
            for i in commodities
            for l in Warehouses
        ) >= nutrient_req[m][n], f"NutrientRequirement_{m}_{n}"

########################### Solve Model ########################################

# Solve the optimization problem
lp.solve(pulp.PULP_CBC_CMD(msg=True))

# Print non-zero decision variables and objective value
for var in lp.variables():
    if var.varValue > 0:
        print(f"{var.name} = {var.varValue}")

print(f"Total Cost = {pulp.value(lp.objective)}")

print(cij['BRAZIL']['BEANS'])