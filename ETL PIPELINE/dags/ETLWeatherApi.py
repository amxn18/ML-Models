from airflow import DAG
from airflow.providers.http.hooks.http import HttpHook 
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.decorators import task
from airflow.utils.dates import days_ago 
import requests
import json 

LATITUDE = '28.65195'
LONGITUDE = '77.23149'
POSTGRES_CONN_ID = 'postgres_default'
API_CONN_ID = 'open_meteo_api'

default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
}

# Defining the DAG
with DAG(
    dag_id = 'etl_weather_api',
    default_args = default_args,
    schedule_interval = '@daily',
    catchup = False) as dags:

    @task()
    def ExtractWeatherData():
        # HttpHook to connect to the API
        httpHook = HttpHook(method='GET', http_conn_id=API_CONN_ID)
        # https://api.open-meteo.com/v1/forecast?latitude=28.65195&longitude=77.23149&current_weather=true
        endpoint = f'/v1/forecast?latitude={LATITUDE}&longitude={LONGITUDE}&current_weather=true'

        response = httpHook.run(endpoint)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API request failed with status code {response.status_code}")

    @task()
    def transformWeatherData(weather_data):
        current_weather = weather_data.get('current_weather', {})
        transformed_data = {
            'latitude': weather_data.get('latitude'),
            'longitude': weather_data.get('longitude'),
            'temperature': current_weather.get('temperature'),
            'windspeed': current_weather.get('windspeed'),
            'winddirection': current_weather.get('winddirection'),
            'weathercode': current_weather.get('weathercode'),
            'time': current_weather.get('time')
        }
        return transformed_data
    
    @task()
    def loadWeatherData(transformed_data):
        postgresHook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        conn = postgresHook.get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS weather_data (
                id SERIAL PRIMARY KEY,
                latitude FLOAT,
                longitude FLOAT,
                temperature FLOAT,
                windspeed FLOAT,
                winddirection FLOAT,
                weathercode INT,
                time TIMESTAMP
            );""")

        insert_query = """
            INSERT INTO weather_data (latitude, longitude, temperature, windspeed, winddirection, weathercode, time)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
        """
        cursor.execute(insert_query, (
            transformed_data['latitude'],
            transformed_data['longitude'],
            transformed_data['temperature'],
            transformed_data['windspeed'],
            transformed_data['winddirection'],
            transformed_data['weathercode'],
            transformed_data['time']
        ))       

        conn.commit()
        cursor.close()

    # DAG Workflow
    weather_data = ExtractWeatherData()
    transformed_data = transformWeatherData(weather_data)
    loadWeatherData(transformed_data)

    
    

