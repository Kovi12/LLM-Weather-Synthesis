import pandas as pd
import requests
import json
import os
import replicate


meteo_api_query = "https://api.open-meteo.com/v1/forecast?latitude=44.4323&longitude=26.1063&hourly=temperature_2m,precipitation_probability&timezone=auto&forecast_days=1"
response = requests.get(meteo_api_query)
data = response.json()


with open('api_data.json', 'w') as json_file:
    json.dump(data, json_file)


table = pd.DataFrame.from_dict(data["hourly"])


query = f"How is the weather today and what should I wear considering I will go to the dentist at 11 am and then go to work until 6 pm?"


csv_text = table[3:25:4].to_csv()
json_text = json.dumps(table[3:25:4].to_dict())
markdown_text = f"# Weather Forecast\n\n{table[3:25:4].to_markdown()}"
html_text = f"<h1>Weather Forecast</h1>{table[3:25:4].to_html()}"

input_text = f"I will ask a question about weather based on the provided data from a reliable weather forecast source.\n\n"

formats = {
    "CSV": csv_text,
    "JSON": json_text,
    "Markdown": markdown_text,
    "HTML": html_text
}

for format_name, text in formats.items():
    full_input_text = f"{input_text} Format: {format_name}\n\n{query}\n\n{format_name} Data:\n{text}"
    print(text)
    os.environ["REPLICATE_API_TOKEN"] = os.environ["REPLICATE_API_TOKEN"]
    api = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])
    output = api.run(
        "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
        input={
            "prompt": full_input_text,
            "max_new_tokens": 1000
        }
    )
    print(f"Format: {format_name}\n")
    for item in output:
        print(item, end="")
    print("\n\n" + "=" * 50 + "\n")
