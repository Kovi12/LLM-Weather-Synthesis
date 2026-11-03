import pandas as pd
import requests
import json
import os
import replicate


meteo_api_query = "https://api.open-meteo.com/v1/forecast?latitude=44.4323&longitude=26.1063&hourly=temperature_2m,precipitation_probability&timezone=auto&forecast_days=14"
response = requests.get(meteo_api_query)
data = response.json()


with open('api_data.json', 'w') as json_file:
    json.dump(data, json_file)


table = pd.DataFrame.from_dict(data["hourly"])






json_text = json.dumps(table[3:200:4].to_dict())
print(json_text)

input_text = f"I will ask a question about weather based on the provided data from a reliable weather forecast source.\n\n"

queries = {
    "Dentist Planning": "How is the weather today and what should I wear considering I will go to the dentist at 11 am and then go to work until 6 pm?",
    "Basic Weather Inquiry": "What is the weather forecast for tomorrow?",
    "Outdoor Activity Planning": "Is it suitable for hiking this weekend?",
    "Event Planning": "Will it rain during my outdoor wedding ceremony at 3 PM on Saturday?",
    "Commute Planning": "Should I expect any weather-related delays for my morning commute tomorrow?",
    "Clothing Recommendation": "What should I wear for a casual outdoor brunch at 12 PM tomorrow?",
    "Health and Safety Concerns": "Is the air quality safe for outdoor exercise this afternoon?",
    "Emergency Preparedness": "What should I do to prepare for an upcoming storm warning?"
}


for query_title, query in queries.items():
    full_input_text = f"{input_text} Query Title: {query_title}\n\n{query}\n\n Data:\n{json_text}"
    print(full_input_text)
    os.environ["REPLICATE_API_TOKEN"] = os.environ["REPLICATE_API_TOKEN"]
    api = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])
    output = api.run(
        "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
        input={
            "prompt": full_input_text,
            "max_new_tokens": 1000
        }
    )
    print(f"Query Title: {query_title}\n")
    for item in output:
        print(item, end="")
    print("\n\n" + "=" * 50 + "\n")
