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
port_data = data['Nodes'][data['Nodes']['LocationTYpe'] == 'Port'][['Location', 'Port capacity (mt/month)']]

hp = port_data.set_index('Location')['Port capacity (mt/month)'].to_dict()

port_to_warehouse = data['LandTransport'].iloc[:11]  
warehouse_to_village = data['LandTransport'].iloc[11:]  


clk = {}

for _, row in port_to_warehouse.iterrows():
    origin = row['Origin']
    destination = row['Destination']
    cost = row['Landtransport cost ($/ton)']
    
    # Initialize the nested dictionary if not already present
    if origin not in clk:
        clk[origin] = {}
    
    clk[origin][destination] = cost

# Extracting warehouse handling costs
warehouse_data = data['Nodes'][data['Nodes']['LocationTYpe'] == 'Warehouse'][['Location', 'Handling cost ($/ton)']]

# Create a dictionary with Warehouse names as keys and their corresponding handling costs as values
hw = warehouse_data.set_index('Location')['Handling cost ($/ton)'].to_dict()

ckm = {}

for _, row in warehouse_to_village.iterrows():
    origin = row['Origin']
    destination = row['Destination']
    cost = row['Landtransport cost ($/ton)']
    
    # Initialize the nested dictionary if not already present
    if origin not in ckm:
        ckm[origin] = {}
    
    ckm[origin][destination] = cost


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
     + pulp.lpSum(hp[port] * pulp.lpSum(y_ijk[commodity][supplier][port] 
                    for commodity in commodities 
                    for supplier in Suppliers) 
                for port in Ports) \
     + pulp.lpSum(clk.get(port, {}).get(warehouse, 0) * z_ikl[commodity][port][warehouse]
                 for commodity in commodities
                 for port in Ports
                 for warehouse in Warehouses) \
    + pulp.lpSum(hw[warehouse] * pulp.lpSum(z_ikl[commodity][port][warehouse] 
                    for port in Ports 
                    for commodity in commodities) 
                for warehouse in Warehouses) \
    + pulp.lpSum(ckm.get(warehouse, {}).get(village, 0) * w_ilm[commodity][warehouse][village]
                 for commodity in commodities 
                 for warehouse in Warehouses
                 for village in Beneficiary_camps)

