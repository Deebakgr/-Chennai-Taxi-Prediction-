import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time

# List of pincodes in Chennai
chennai_pin_codes = [
    600001, 600002, 600003, 600004, 600005, 600006, 600007, 600008, 600009, 600010,
    600011, 600012, 600013, 600014, 600015, 600016, 600017, 600018, 600019, 600020,
    600021, 600022, 600023, 600024, 600025, 600026, 600027, 600028, 600029, 600030,
    600031, 600032, 600033, 600034, 600035, 600036, 600037, 600038, 600039, 600040,
    600041, 600042, 600043, 600044, 600045, 600046, 600047, 600048, 600049, 600050,
    600051, 600052, 600053, 600054, 600055, 600056, 600057, 600058, 600059, 600060,
    600061, 600062, 600063, 600064, 600065, 600066, 600067, 600068, 600069, 600070,
    600071, 600072, 600073, 600074, 600075, 600076, 600077, 600078, 600079, 600080,
    600081, 600082, 600083, 600084, 600085, 600086, 600087, 600088, 600089, 600090,
    600091, 600092, 600093, 600094, 600095, 600096, 600097, 600098, 600099, 600100
]

# Initialize geolocator
geolocator = Nominatim(user_agent="chennai_geocoder")

# Function to get latitude and longitude for a pincode
def get_lat_lon(pincode):
    try:
        location = geolocator.geocode(f"{pincode}, Chennai, India")
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except GeocoderTimedOut:
        return get_lat_lon(pincode)

# Create a DataFrame to store the results
pincode_df = pd.DataFrame(chennai_pin_codes, columns=['pincode'])
pincode_df['latitude'] = None
pincode_df['longitude'] = None

# Get latitude and longitude for each pincode
for idx, row in pincode_df.iterrows():
    lat, lon = get_lat_lon(row['pincode'])
    pincode_df.at[idx, 'latitude'] = lat
    pincode_df.at[idx, 'longitude'] = lon
    time.sleep(1)  # Sleep to respect the API rate limit

# Save the results to a new CSV file
pincode_df.to_csv('chennai_pincode_lat_lon.csv', index=False)

print(pincode_df)
