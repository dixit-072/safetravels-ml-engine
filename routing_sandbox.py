import requests
import polyline # We need this to decode the route path!

def test_route_engine(source_name, source_coords, dest_name, dest_coords):
    print(f"🚗 Calculating route from {source_name} to {dest_name}...\n")
    
    # OSRM requires coordinates in Longitude,Latitude order!
    lon1, lat1 = source_coords
    lon2, lat2 = dest_coords
    
    # The Free OSRM API Endpoint
    osrm_url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
    
    # We ask for 'overview=full' to get the exact path of the road
    params = {"overview": "full"}
    
    response = requests.get(osrm_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        # 1. Get the primary route
        route = data["routes"][0]
        
        # 2. Extract Distance and Time
        distance_km = route["distance"] / 1000  # OSRM returns meters, convert to km
        duration_hours = route["duration"] / 3600 # OSRM returns seconds, convert to hours
        
        print(f"✅ Route Found!")
        print(f"📏 Total Distance: {distance_km:.2f} km")
        print(f"⏱️ Estimated Driving Time: {duration_hours:.2f} hours\n")
        
        # 3. Decode the geometry into actual GPS Checkpoints
        encoded_path = route["geometry"]
        checkpoints = polyline.decode(encoded_path)
        
        print(f"🗺️ The routing engine generated {len(checkpoints)} GPS coordinates along the highway.")
        
        # 4. Find the exact halfway point!
        halfway_index = len(checkpoints) // 2
        halfway_coord = checkpoints[halfway_index]
        
        print(f"📍 Start: {checkpoints[0]} ({source_name})")
        print(f"📍 EXACT HALFWAY CHECKPOINT: {halfway_coord}")
        print(f"📍 End: {checkpoints[-1]} ({dest_name})")
        
    else:
        print("❌ Routing failed:", response.text)

if __name__ == "__main__":
    # Let's test a road trip from Delhi to Manali
    DELHI = (77.2090, 28.6139)   # Longitude, Latitude
    MANALI = (77.1887, 32.2396)  # Longitude, Latitude
    
    test_route_engine("New Delhi", DELHI, "Manali", MANALI)