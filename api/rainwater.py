from rest_framework.response import Response 
import requests
from rest_framework.decorators import api_view
from django.http import JsonResponse
import openmeteo_requests
from datetime import datetime, timedelta
import requests_cache
import pandas as pd
from retry_requests import retry
from dateutil.relativedelta import relativedelta   

# lat=13.032166558520242
# lon=77.6292543642592



# area=111.4836

today=datetime.now()-timedelta(days=2)
end=str(today.strftime("%Y-%m-%d"))
d1=str(today.strftime("%Y/%m/%d"))

dt=datetime.now()-relativedelta(years=1)
start=str(dt.strftime("%Y-%m-%d"))
d2=str(dt.strftime("%Y/%m/%d"))

date_format = "%Y/%m/%d"

a = datetime.strptime(d1, date_format)
b = datetime.strptime(d2, date_format)

delta = a-b


def rainfall(lat, long, area):
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)



    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": long,
        "start_date": start,
        "end_date": end,
        "daily": "rain_sum",
        "timezone": "auto"
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_rain_sum = daily.Variables(0).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
        end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
    )}
    daily_data["rain_sum"] = daily_rain_sum

    daily_dataframe = pd.DataFrame(data = daily_data)
    sum=0
    for ind in daily_dataframe.index:
        sum+=daily_dataframe['rain_sum'][ind]

    

    #thornshit constant
    k=1.02

    #get mean temp
    mean_temperature=temp(lat, long)

    #evaporation loss
    evap_loss=k*(mean_temperature/5)**1.514

    #Effective rainfall=rainfall(fetched from api)*(1-evaporation loss-absorption loss-other losses)
    eff_rainfall=sum*(1-evap_loss/100-0.3005)

    harvested=eff_rainfall*area
    

    
    usage_litres=32000
    connection_type='domestic'

    def calculate_water_charges_domestic(usage_liters):
        if usage_liters <= 8000:
            water_tariff = 7
            sanitary_charge = 14
            meter_cost = 100
        elif 8001 <= usage_liters <= 25000:
            water_tariff = 11
            sanitary_charge = 0.25 * water_tariff
            meter_cost = 30
        elif 25001 <= usage_liters <= 50000:
            water_tariff = 26
            sanitary_charge = 0.25 * water_tariff
            meter_cost = 50
        else:
            water_tariff = 45
            sanitary_charge = 0.25 * water_tariff
            meter_cost = 150
        
        water_charge = water_tariff * (usage_liters / 1000)
        total_charge = water_charge + sanitary_charge + meter_cost
        return total_charge

    def calculate_water_charges_non_domestic(usage_liters):
        if usage_liters <= 10000:
            water_tariff = 50
            sanitary_charge = 0.25 * water_tariff
            meter_cost = 50
        elif 10001 <= usage_liters <= 25000:
            water_tariff = 57
            sanitary_charge = 0
            meter_cost = 75
        elif 25001 <= usage_liters <= 50000:
            water_tariff = 65
            sanitary_charge = 0
            meter_cost = 100
        elif 50001 <= usage_liters <= 75000:
            water_tariff = 76
            sanitary_charge = 0
            meter_cost = 125
        else:
            water_tariff = 87
            sanitary_charge = 0
            meter_cost = 175
        
        water_charge = water_tariff * (usage_liters / 1000)
        total_charge = water_charge + sanitary_charge + meter_cost
        return total_charge

    def main(litres):
        
        if connection_type == "domestic":
            total_charge = calculate_water_charges_domestic(litres)
        else :
            total_charge = calculate_water_charges_non_domestic(litres)
        
        return total_charge

    net_usage=usage_litres-(harvested/12)
    saving=main(usage_litres)-main(net_usage)


    return [harvested, usage_litres, main(usage_litres), saving]


def temp(lat, long):
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": long,
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_mean",
        "timezone": "auto"
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    daily = response.Daily()
    daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
        end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
    )}
    daily_data["temperature_2m_mean"] = daily_temperature_2m_mean

    daily_dataframe = pd.DataFrame(data = daily_data)

    sum=0
    for ind in daily_dataframe.index:
        sum+=daily_dataframe['temperature_2m_mean'][ind]
    return (sum/int(delta.days))




# @api_view(['GET'])
# def getData(request):
#     dic={
#         'latitude': lat,
#         'longitude': lon,
#         'area': area,
#         'rain_water_harvested':rainfall()[0],
#         'litres_consumed':rainfall()[1],
#         'bill_amount':rainfall()[2],
#         'monthly_saving': rainfall()[3],
#     }
#     response = JsonResponse(dic, status=200)
#     return response