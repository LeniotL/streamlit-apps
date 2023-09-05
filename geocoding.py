
import requests

def get_coordinates_from_address(address):
    API_KEY = "lgJni2MDMSvRTqVguba17QMqzjMmwkWx"
    url = f"https://geloky.com/api/geo/geocode?address={address}&key={API_KEY}&format=geloky"

    TOKYO_LAT = 35.6895
    TOKYO_LONG = 139.6917

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()

        # Check if the list is empty
        if not data:
            print(f"No matching results for address: {address}")
            return TOKYO_LAT, TOKYO_LONG  # Default Tokyo coordinates

        result = data[0]  # Access the first dictionary in the list
        latitude = result["latitude"]
        longitude = result["longitude"]
        return latitude, longitude
    else:
        print(f"Request failed with status code {response.status_code}")
        return TOKYO_LAT, TOKYO_LONG  # Default Tokyo coordinates in case of an API failure
