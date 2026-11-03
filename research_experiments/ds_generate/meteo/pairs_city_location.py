from ds_generate.general.tools.cities import *
from ds_generate.meteo.times_experssions import *

import random
import datetime

def random_date():
    year = random.randint(2000, 2024)
    start_date = datetime.datetime(year, 1, 1)
    end_date = datetime.datetime(year, 12, 31)
    days_between = (end_date - start_date).days
    random_days = random.randint(0, days_between)
    random_hour_minute = random.randint(0, 23), random.randint(0, 59)
    rd = start_date + datetime.timedelta(
        days=random_days, 
        hours=random_hour_minute[0],
        minutes=random_hour_minute[1]
    )
    return rd

def generate_city_location_pairs():
    # pick 1, 2, or 3 cities, with 40%, 30%, 30% chance respectively
    num_cities = random.choices([1, 2, 3], [0.4, 0.3, 0.3])[0] 
    cities_choice = get_cities(num_cities).reset_index(drop=True)

    # 30% chance of same time, 70% chance of different times
    same_time = random.choices([True, False], [0.3, 0.7])[0] 
    if same_time:
        time = get_time(1)[0]
        label = [(cities_choice[CITY_COL][i], cities_choice[COUNTRY_COL][i], time) for i in range(num_cities)]
    else:
        time = get_time(num_cities)
        label = [(cities_choice[CITY_COL][i], cities_choice[COUNTRY_COL][i], time[i]) for i in range(num_cities)]
    
    text = ""
    i = 1
    for row in label:
        text += f"City {i}:\n\t- city, country: {row[0]}, {row[1]}\n\t- temporal expression: {row[2]}\n"
        i += 1
    ref = random_date().strftime("%A, %d %B %Y, %I:%M %p")
    text += f"Reference date: {ref}\n"
    return text, ref
