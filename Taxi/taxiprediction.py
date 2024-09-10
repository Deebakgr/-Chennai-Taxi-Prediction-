









# import streamlit as st
# import pandas as pd
# import numpy as np
# import folium
# from folium.plugins import MarkerCluster
# import streamlit.components.v1 as components
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from geopy.distance import geodesic

# # Load dataset
# @st.cache_data
# def load_data(file_path):
#     return pd.read_excel(file_path)

# # Load pincode latitude and longitude data
# @st.cache_data
# def load_pincode_data(file_path):
#     return pd.read_csv(file_path)

# # Function to create a map with markers and lines
# def create_map(from_location, to_location, distance, duration, from_pincode, to_pincode):
#     chennai_map = folium.Map(location=[13.0827, 80.2707], zoom_start=12)
#     marker_cluster = MarkerCluster().add_to(chennai_map)
    
#     # Add markers
#     folium.Marker(
#         location=from_location,
#         popup=f"From: {from_pincode}\nDistance: {distance:.2f} km\nDuration: {duration:.2f} mins",
#         icon=folium.Icon(color='green', icon='info-sign')
#     ).add_to(marker_cluster)
    
#     folium.Marker(
#         location=to_location,
#         popup=f"To: {to_pincode}\nDistance: {distance:.2f} km\nDuration: {duration:.2f} mins",
#         icon=folium.Icon(color='red', icon='info-sign')
#     ).add_to(marker_cluster)
    
#     # Draw line between locations
#     folium.PolyLine(
#         locations=[from_location, to_location],
#         color='blue',
#         weight=2.5,
#         opacity=1
#     ).add_to(chennai_map)

#     return chennai_map

# # Function to predict fare
# def predict_fare(model, scaler, distance, duration):
#     input_data = pd.DataFrame({'distance': [distance], 'duration': [duration]})
#     input_data_scaled = scaler.transform(input_data)
#     return model.predict(input_data_scaled)[0]

# # Add favicon to the browser tab
# favicon = "taxi-icon-orange-sticker-vector-1727287.jpg"
# st.set_page_config(page_title="Chennai Taxi Fare Predictor", page_icon=favicon)



# # Load data
# df = load_data('chennai_taxi_fares_with_pin_codes.xlsx')
# pincode_df = load_pincode_data('chennai_pincode_lat_lon.csv')

# # Check if 'pin_code' exists in both datasets
# if 'pin_code' in df.columns and 'pincode' in pincode_df.columns:
#     # Merge the latitude and longitude data into the main dataframe
#     df = df.merge(pincode_df, left_on='pin_code', right_on='pincode', how='left', suffixes=('', '_pincode'))

#     # Define features and target
#     features = df[['distance', 'duration']]
#     target = df['fare']

#     # Split data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#     # Feature scaling
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # Hyperparameter tuning using GridSearchCV with reduced parameters for speed
#     param_grid = {
#         'n_estimators': [50, 100],
#         'max_depth': [10, 20, None]
#     }

#     rf_model = RandomForestRegressor(random_state=42)
#     grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
#     grid_search.fit(X_train_scaled, y_train)

#     best_rf_model = grid_search.best_estimator_

#     # Predict using the best model
#     rf_predictions = best_rf_model.predict(X_test_scaled)

#     # Evaluate model
#     rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
#     rf_mae = mean_absolute_error(y_test, rf_predictions)
#     rf_r2 = r2_score(y_test, rf_predictions)

#     # Display metrics
#     st.write(f"Random Forest RMSE: {rf_rmse:.2f}")
#     st.write(f"Random Forest MAE: {rf_mae:.2f}")
#     st.write(f"Random Forest R^2: {rf_r2:.2f}")

#     # Predict Fare from Pincode to Pincode
#     st.subheader('Predict Fare from Pincode to Pincode')

#     from_pincode = st.selectbox('From Pincode', df['pin_code'].unique(), key='from_pincode')
#     to_pincode = st.selectbox('To Pincode', df['pin_code'].unique(), key='to_pincode')

#     from_location = pincode_df[pincode_df['pincode'] == from_pincode][['latitude', 'longitude']].values
#     to_location = pincode_df[pincode_df['pincode'] == to_pincode][['latitude', 'longitude']].values

#     if from_location.size > 0 and to_location.size > 0:
#         from_location = from_location[0]
#         to_location = to_location[0]

#         # Calculate distance and estimated duration
#         distance = geodesic(from_location, to_location).km
#         duration = (distance / 40) * 60  # Assuming average speed of 40 km/h

#         st.write(f"Estimated Distance: {distance:.2f} km")
#         st.write(f"Estimated Duration: {duration:.2f} mins")

#         if st.button('Predict Fare for Pincode to Pincode'):
#             fare = predict_fare(best_rf_model, scaler, distance, duration)
#             st.write(f"Predicted Fare from Pincode {from_pincode} to Pincode {to_pincode}: {fare:.2f} INR")
            
#             # Create and display map with trip details
#             map_object = create_map(from_location, to_location, distance, duration, from_pincode, to_pincode)
#             map_path = "chennai_trip_map.html"
#             map_object.save(map_path)

#             with open(map_path, 'r') as f:
#                 map_html = f.read()
#             components.html(map_html, height=600, width=800)
# else:
#     st.write("One of the datasets is missing the required 'pin_code' or 'pincode' column.")
















# import streamlit as st
# import pandas as pd
# import numpy as np
# from geopy.distance import geodesic
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import requests

# # Load dataset
# @st.cache_data
# def load_data(file_path):
#     return pd.read_excel(file_path)

# # Load pincode latitude and longitude data
# @st.cache_data
# def load_pincode_data(file_path):
#     return pd.read_csv(file_path)

# # Function to predict fare
# def predict_fare(model, scaler, distance, duration):
#     input_data = pd.DataFrame({'distance': [distance], 'duration': [duration]})
#     input_data_scaled = scaler.transform(input_data)
#     return model.predict(input_data_scaled)[0]

# # Add favicon to the browser tab
# favicon = "taxi-icon-orange-sticker-vector-1727287.jpg"
# st.set_page_config(page_title="Chennai Taxi Fare Predictor", page_icon=favicon)

# # Display logo in the app
# st.image(favicon, use_column_width=True)

# # Load data
# df = load_data('chennai_taxi_fares_with_pin_codes.xlsx')
# pincode_df = load_pincode_data('chennai_pincode_lat_lon.csv')

# # Check if 'pin_code' exists in both datasets
# if 'pin_code' in df.columns and 'pincode' in pincode_df.columns:
#     # Merge the latitude and longitude data into the main dataframe
#     df = df.merge(pincode_df, left_on='pin_code', right_on='pincode', how='left', suffixes=('', '_pincode'))

#     # Define features and target
#     features = df[['distance', 'duration']]
#     target = df['fare']

#     # Split data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#     # Feature scaling
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # Hyperparameter tuning using GridSearchCV with reduced parameters for speed
#     param_grid = {
#         'n_estimators': [50, 100],
#         'max_depth': [10, 20, None]
#     }

#     rf_model = RandomForestRegressor(random_state=42)
#     grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
#     grid_search.fit(X_train_scaled, y_train)

#     best_rf_model = grid_search.best_estimator_

#     # Predict using the best model
#     rf_predictions = best_rf_model.predict(X_test_scaled)

#     # Evaluate model
#     rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
#     rf_mae = mean_absolute_error(y_test, rf_predictions)
#     rf_r2 = r2_score(y_test, rf_predictions)

#     # Display metrics
#     st.write(f"Random Forest RMSE: {rf_rmse:.2f}")
#     st.write(f"Random Forest MAE: {rf_mae:.2f}")
#     st.write(f"Random Forest R^2: {rf_r2:.2f}")

#     # Predict Fare from Pincode to Pincode
#     st.subheader('Predict Fare from Pincode to Pincode')

#     from_pincode = st.selectbox('From Pincode', df['pin_code'].unique(), key='from_pincode')
#     to_pincode = st.selectbox('To Pincode', df['pin_code'].unique(), key='to_pincode')

#     from_location = pincode_df[pincode_df['pincode'] == from_pincode][['latitude', 'longitude']].values
#     to_location = pincode_df[pincode_df['pincode'] == to_pincode][['latitude', 'longitude']].values

#     if from_location.size > 0 and to_location.size > 0:
#         from_location = from_location[0]
#         to_location = to_location[0]

#         # Generate Google Maps URL
#         google_maps_url = f"https://www.google.com/maps/dir/?api=1&origin={from_location[0]},{from_location[1]}&destination={to_location[0]},{to_location[1]}&travelmode=driving"

#         # Display link to Google Maps
#         st.markdown(f'[View Route on Google Maps]({google_maps_url})')

#         # Display inputs for distance and duration
#         st.write("Enter the distance and duration from Google Maps manually:")
#         distance = st.number_input("Distance (km)", min_value=0.0, format="%.2f")
#         duration = st.number_input("Duration (minutes)", min_value=0.0, format="%.2f")

#         if st.button('Predict Fare for Pincode to Pincode'):
#             if distance > 0 and duration > 0:
#                 fare = predict_fare(best_rf_model, scaler, distance, duration)
#                 st.write(f"Predicted Fare from Pincode {from_pincode} to Pincode {to_pincode}: {fare:.2f} INR")
# else:
#     st.write("One of the datasets is missing the required 'pin_code' or 'pincode' column.")













# import streamlit as st
# import pandas as pd
# import numpy as np
# from geopy.distance import geodesic
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# # Load dataset
# @st.cache_data
# def load_data(file_path):
#     return pd.read_excel(file_path)

# # Load pincode latitude and longitude data
# @st.cache_data
# def load_pincode_data(file_path):
#     return pd.read_csv(file_path)

# # Function to predict fare
# def predict_fare(model, scaler, distance, duration):
#     input_data = pd.DataFrame({'distance': [distance], 'duration': [duration]})
#     input_data_scaled = scaler.transform(input_data)
#     return model.predict(input_data_scaled)[0]

# # Function to calculate distance and estimated duration
# def calculate_distance_duration(from_location, to_location):
#     # Calculate geodesic distance
#     distance = geodesic(from_location, to_location).km

#     # Assume an average speed of 30 km/h to estimate duration
#     duration = (distance / 30) * 60  # Convert hours to minutes

#     return distance, duration

# # Add favicon to the browser tab
# favicon = "taxi-icon-orange-sticker-vector-1727287.jpg"
# st.set_page_config(page_title="Chennai Taxi Fare Predictor", page_icon=favicon)


# # Load data
# df = load_data('chennai_taxi_fares_with_pin_codes.xlsx')
# pincode_df = load_pincode_data('chennai_pincode_lat_lon.csv')

# # Check if 'pin_code' exists in both datasets
# if 'pin_code' in df.columns and 'pincode' in pincode_df.columns:
#     # Merge the latitude and longitude data into the main dataframe
#     df = df.merge(pincode_df, left_on='pin_code', right_on='pincode', how='left', suffixes=('', '_pincode'))

#     # Define features and target
#     features = df[['distance', 'duration']]
#     target = df['fare']

#     # Split data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#     # Feature scaling
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # Hyperparameter tuning using GridSearchCV with reduced parameters for speed
#     param_grid = {
#         'n_estimators': [50, 100],
#         'max_depth': [10, 20, None]
#     }

#     rf_model = RandomForestRegressor(random_state=42)
#     grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
#     grid_search.fit(X_train_scaled, y_train)

#     best_rf_model = grid_search.best_estimator_

#     # Predict using the best model
#     rf_predictions = best_rf_model.predict(X_test_scaled)

#     # Evaluate model
#     rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
#     rf_mae = mean_absolute_error(y_test, rf_predictions)
#     rf_r2 = r2_score(y_test, rf_predictions)

#     # Display metrics
#     st.write(f"Random Forest RMSE: {rf_rmse:.2f}")
#     st.write(f"Random Forest MAE: {rf_mae:.2f}")
#     st.write(f"Random Forest R^2: {rf_r2:.2f}")

#     # Predict Fare from Pincode to Pincode
#     st.subheader('Predict Fare from Pincode to Pincode')

#     from_pincode = st.selectbox('From Pincode', df['pin_code'].unique(), key='from_pincode')
#     to_pincode = st.selectbox('To Pincode', df['pin_code'].unique(), key='to_pincode')

#     from_location = pincode_df[pincode_df['pincode'] == from_pincode][['latitude', 'longitude']].values
#     to_location = pincode_df[pincode_df['pincode'] == to_pincode][['latitude', 'longitude']].values

#     if from_location.size > 0 and to_location.size > 0:
#         from_location = from_location[0]
#         to_location = to_location[0]

#         # Calculate distance and estimated duration
#         distance, duration = calculate_distance_duration(from_location, to_location)

#         st.write(f"Estimated Distance: {distance:.2f} km")
#         st.write(f"Estimated Duration: {duration:.2f} mins")

#         # Generate Google Maps URL
#         google_maps_url = f"https://www.google.com/maps/dir/?api=1&origin={from_location[0]},{from_location[1]}&destination={to_location[0]},{to_location[1]}&travelmode=driving"

#         # Display link to Google Maps
#         st.markdown(f'[View Route on Google Maps]({google_maps_url})')

#         if st.button('Predict Fare for Pincode to Pincode'):
#             fare = predict_fare(best_rf_model, scaler, distance, duration)
#             st.write(f"Predicted Fare from Pincode {from_pincode} to Pincode {to_pincode}: {fare:.2f} INR")
# else:
#     st.write("One of the datasets is missing the required 'pin_code' or 'pincode' column.")







# import streamlit as st
# import pandas as pd
# import numpy as np
# from geopy.distance import geodesic
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# # Load dataset
# @st.cache_data
# def load_data(file_path):
#     return pd.read_excel(file_path)

# # Load pincode latitude and longitude data
# @st.cache_data
# def load_pincode_data(file_path):
#     return pd.read_csv(file_path)

# # Function to predict fare
# def predict_fare(model, scaler, distance, duration):
#     input_data = pd.DataFrame({'distance': [distance], 'duration': [duration]})
#     input_data_scaled = scaler.transform(input_data)
#     return model.predict(input_data_scaled)[0]

# # Add favicon to the browser tab
# favicon = "taxi-icon-orange-sticker-vector-1727287.jpg"
# st.set_page_config(page_title="Chennai Taxi Fare Predictor", page_icon=favicon)



# # Load data
# df = load_data('chennai_taxi_fares_with_pin_codes.xlsx')
# pincode_df = load_pincode_data('chennai_pincode_lat_lon.csv')

# # Check if 'pin_code' exists in both datasets
# if 'pin_code' in df.columns and 'pincode' in pincode_df.columns:
#     # Merge the latitude and longitude data into the main dataframe
#     df = df.merge(pincode_df, left_on='pin_code', right_on='pincode', how='left', suffixes=('', '_pincode'))

#     # Define features and target
#     features = df[['distance', 'duration']]
#     target = df['fare']

#     # Split data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#     # Feature scaling
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # Hyperparameter tuning using GridSearchCV with reduced parameters for speed
#     param_grid = {
#         'n_estimators': [50, 100],
#         'max_depth': [10, 20, None]
#     }

#     rf_model = RandomForestRegressor(random_state=42)
#     grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
#     grid_search.fit(X_train_scaled, y_train)

#     best_rf_model = grid_search.best_estimator_

#     # Predict using the best model
#     rf_predictions = best_rf_model.predict(X_test_scaled)

#     # Evaluate model
#     rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
#     rf_mae = mean_absolute_error(y_test, rf_predictions)
#     rf_r2 = r2_score(y_test, rf_predictions)

#     # Display metrics
#     st.write(f"Random Forest RMSE: {rf_rmse:.2f}")
#     st.write(f"Random Forest MAE: {rf_mae:.2f}")
#     st.write(f"Random Forest R^2: {rf_r2:.2f}")

#     # Predict Fare from Pincode to Pincode
#     st.subheader('Predict Fare from Pincode to Pincode')

#     from_pincode = st.selectbox('From Pincode', df['pin_code'].unique(), key='from_pincode')
#     to_pincode = st.selectbox('To Pincode', df['pin_code'].unique(), key='to_pincode')

#     from_location = pincode_df[pincode_df['pincode'] == from_pincode][['latitude', 'longitude']].values
#     to_location = pincode_df[pincode_df['pincode'] == to_pincode][['latitude', 'longitude']].values

#     if from_location.size > 0 and to_location.size > 0:
#         from_location = from_location[0]
#         to_location = to_location[0]

#         # Generate Google Maps URL
#         google_maps_url = f"https://www.google.com/maps/dir/?api=1&origin={from_location[0]},{from_location[1]}&destination={to_location[0]},{to_location[1]}&travelmode=driving"

#         # Display link to Google Maps
#         st.markdown(f'[View Route on Google Maps]({google_maps_url})')

#         # Manual input for distance and duration
#         distance = st.number_input('Enter the distance (in km)', min_value=0.0, format="%.2f")
#         duration = st.number_input('Enter the duration (in minutes)', min_value=0.0, format="%.2f")

#         if distance > 0 and duration > 0:
#             st.write(f"Entered Distance: {distance:.2f} km")
#             st.write(f"Entered Duration: {duration:.2f} mins")

#             if st.button('Predict Fare for Pincode to Pincode'):
#                 fare = predict_fare(best_rf_model, scaler, distance, duration)
#                 st.write(f"Predicted Fare from Pincode {from_pincode} to Pincode {to_pincode}: {fare:.2f} INR")
# else:
#     st.write("One of the datasets is missing the required 'pin_code' or 'pincode' column.")






# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# # Load dataset
# @st.cache_data
# def load_data(file_path):
#     return pd.read_excel(file_path)

# # Load pincode latitude and longitude data
# @st.cache_data
# def load_pincode_data(file_path):
#     return pd.read_csv(file_path)

# # Function to predict fare
# def predict_fare(model, scaler, distance, duration, time_of_day, traffic_condition):
#     # Map time of day and traffic condition to numerical values
#     time_mapping = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
#     traffic_mapping = {'Light': 0, 'Moderate': 1, 'Heavy': 2}

#     input_data = pd.DataFrame({
#         'distance': [distance],
#         'duration': [duration],
#         'time_of_day': [time_mapping[time_of_day]],
#         'traffic_condition': [traffic_mapping[traffic_condition]]
#     })
#     input_data_scaled = scaler.transform(input_data)
#     return model.predict(input_data_scaled)[0]

# # Add favicon to the browser tab
# favicon = "taxi-icon-orange-sticker-vector-1727287.jpg"
# st.set_page_config(page_title="Chennai Taxi Fare Predictor", page_icon=favicon)

# # Display logo in the app
# st.image(favicon, use_column_width=True)

# # Load data
# df = load_data('chennai_taxi_fares_with_pin_codes.xlsx')
# pincode_df = load_pincode_data('chennai_pincode_lat_lon.csv')

# # Check if 'pin_code' exists in both datasets
# if 'pin_code' in df.columns and 'pincode' in pincode_df.columns:
#     # Merge the latitude and longitude data into the main dataframe
#     df = df.merge(pincode_df, left_on='pin_code', right_on='pincode', how='left', suffixes=('', '_pincode'))

#     # Add time_of_day and traffic_condition columns manually
#     df['time_of_day'] = np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], size=len(df))
#     df['traffic_condition'] = np.random.choice(['Light', 'Moderate', 'Heavy'], size=len(df))

#     # Map time_of_day and traffic_condition to numerical values for model training
#     time_mapping = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
#     traffic_mapping = {'Light': 0, 'Moderate': 1, 'Heavy': 2}

#     df['time_of_day'] = df['time_of_day'].map(time_mapping)
#     df['traffic_condition'] = df['traffic_condition'].map(traffic_mapping)

#     # Define features and target
#     features = df[['distance', 'duration', 'time_of_day', 'traffic_condition']]
#     target = df['fare']

#     # Split data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#     # Feature scaling
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # Hyperparameter tuning using GridSearchCV with refined parameters
#     param_grid = {
#         'n_estimators': [50, 100, 200],
#         'max_depth': [10, 20, 30, None],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4]
#     }

#     rf_model = RandomForestRegressor(random_state=42)
#     grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
#     grid_search.fit(X_train_scaled, y_train)

#     best_rf_model = grid_search.best_estimator_

#     # Predict using the best model
#     rf_predictions = best_rf_model.predict(X_test_scaled)

#     # Evaluate model
#     rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
#     rf_mae = mean_absolute_error(y_test, rf_predictions)
#     rf_r2 = r2_score(y_test, rf_predictions)

#     # Display metrics
#     st.write(f"Random Forest RMSE: {rf_rmse:.2f}")
#     st.write(f"Random Forest MAE: {rf_mae:.2f}")
#     st.write(f"Random Forest R^2: {rf_r2:.2f}")

#     # Predict Fare from Pincode to Pincode
#     st.subheader('Predict Fare from Pincode to Pincode')

#     from_pincode = st.selectbox('From Pincode', df['pin_code'].unique(), key='from_pincode')
#     to_pincode = st.selectbox('To Pincode', df['pin_code'].unique(), key='to_pincode')

#     from_location = pincode_df[pincode_df['pincode'] == from_pincode][['latitude', 'longitude']].values
#     to_location = pincode_df[pincode_df['pincode'] == to_pincode][['latitude', 'longitude']].values

#     if from_location.size > 0 and to_location.size > 0:
#         from_location = from_location[0]
#         to_location = to_location[0]

#         # Generate Google Maps URL
#         google_maps_url = f"https://www.google.com/maps/dir/?api=1&origin={from_location[0]},{from_location[1]}&destination={to_location[0]},{to_location[1]}&travelmode=driving"

#         # Display link to Google Maps
#         st.markdown(f'[View Route on Google Maps]({google_maps_url})')

#         st.write("Please open the link above, check the shortest distance and estimated duration on Google Maps, and enter them below.")

#         # Manual input for distance and duration
#         distance = st.number_input('Enter the distance (in km)', min_value=0.0, format="%.2f")
#         duration = st.number_input('Enter the duration (in minutes)', min_value=0.0, format="%.2f")

#         # Manual input for time of day
#         time_of_day = st.selectbox('Select the time of day', ['Morning', 'Afternoon', 'Evening', 'Night'])

#         # Manual input for traffic condition
#         traffic_condition = st.selectbox('Select the traffic condition', ['Light', 'Moderate', 'Heavy'])

#         if distance > 0 and duration > 0:
#             st.write(f"Entered Distance: {distance:.2f} km")
#             st.write(f"Entered Duration: {duration:.2f} mins")

#             if st.button('Predict Fare for Pincode to Pincode'):
#                 fare = predict_fare(best_rf_model, scaler, distance, duration, time_of_day, traffic_condition)
#                 st.write(f"Predicted Fare from Pincode {from_pincode} to Pincode {to_pincode}: {fare:.2f} INR")
# else:
#     st.write("One of the datasets is missing the required 'pin_code' or 'pincode' column.")









import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime

# Load dataset
@st.cache_data
def load_data(file_path):
    return pd.read_excel(file_path)

# Load pincode latitude and longitude data
@st.cache_data
def load_pincode_data(file_path):
    return pd.read_csv(file_path)

# Train the model with reduced time complexity
@st.cache_resource
def train_model(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20, None]
    }

    rf_model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_

# Function to predict fare
def predict_fare(model, scaler, distance, duration, time_of_day, traffic_condition):
    traffic_multiplier = 1.0
    if traffic_condition == 0:  # Light Traffic
        traffic_multiplier = 0.8
    elif traffic_condition == 1:  # Moderate Traffic
        traffic_multiplier = 1.0
    elif traffic_condition == 2:  # Heavy Traffic
        traffic_multiplier = 1.2

    input_data = pd.DataFrame({
        'distance': [distance],
        'duration': [duration],
        'time_of_day': [time_of_day],
        'traffic_condition': [traffic_condition]
    })
    input_data_scaled = scaler.transform(input_data)
    base_fare = model.predict(input_data_scaled)[0]
    
    # Apply traffic multiplier to base fare
    adjusted_fare = base_fare * traffic_multiplier
    return adjusted_fare

# Estimate traffic based on time of day
def estimate_traffic_condition(time_of_day):
    if time_of_day in ['Morning', 'Evening']:
        return 2  # Heavy Traffic
    elif time_of_day == 'Afternoon':
        return 1  # Moderate Traffic
    else:
        return 0  # Light Traffic

# Map time of day to AM/PM format
def map_time_of_day(hour):
    if 7 <= hour < 12:
        return 'Morning'  # AM
    elif 12 <= hour < 16:
        return 'Afternoon'  # PM
    elif 16 <= hour < 20:
        return 'Evening'  # PM
    else:
        return 'Night'  # PM

# Add favicon to the browser tab
favicon = "taxi-icon-orange-sticker-vector-1727287.jpg"
st.set_page_config(page_title="Chennai Taxi Fare Predictor", page_icon=favicon)



# Load data
df = load_data('../Taxi/chennai_taxi_fares_with_pin_codes.xlsx')
pincode_df = load_pincode_data('../Taxi/chennai_pincode_lat_lon.csv')

# Check if 'pin_code' exists in both datasets
if 'pin_code' in df.columns and 'pincode' in pincode_df.columns:
    # Merge the latitude and longitude data into the main dataframe
    df = df.merge(pincode_df, left_on='pin_code', right_on='pincode', how='left', suffixes=('', '_pincode'))

    # Add time_of_day and traffic_condition columns manually
    df['time_of_day'] = np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], size=len(df))
    df['traffic_condition'] = df['time_of_day'].apply(estimate_traffic_condition)

    # Map time_of_day and traffic_condition to numerical values for model training
    time_mapping = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}

    df['time_of_day'] = df['time_of_day'].map(time_mapping)

    # Define features and target
    features = df[['distance', 'duration', 'time_of_day', 'traffic_condition']]
    target = df['fare']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model with optimized settings
    best_rf_model = train_model(X_train_scaled, y_train)

    # Predict using the best model
    rf_predictions = best_rf_model.predict(X_test_scaled)

    # Evaluate model
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
    rf_mae = mean_absolute_error(y_test, rf_predictions)
    rf_r2 = r2_score(y_test, rf_predictions)

   

    # Predict Fare from Pincode to Pincode
    st.subheader('Predict Fare from Pincode to Pincode')

    from_pincode = st.selectbox('From Pincode', df['pin_code'].unique(), key='from_pincode')
    to_pincode = st.selectbox('To Pincode', df['pin_code'].unique(), key='to_pincode')

    from_location = pincode_df[pincode_df['pincode'] == from_pincode][['latitude', 'longitude']].values
    to_location = pincode_df[pincode_df['pincode'] == to_pincode][['latitude', 'longitude']].values

    if from_location.size > 0 and to_location.size > 0:
        from_location = from_location[0]
        to_location = to_location[0]

        # Generate Google Maps URL
        google_maps_url = f"https://www.google.com/maps/dir/?api=1&origin={from_location[0]},{from_location[1]}&destination={to_location[0]},{to_location[1]}&travelmode=driving"

        # Display link to Google Maps
        st.markdown(f'[View Route on Google Maps]({google_maps_url})')

        st.write("Please open the link above, check the shortest distance and estimated duration on Google Maps, and enter them below.")

        # Manual input for distance and duration
        distance = st.number_input('Enter the distance (in km)', min_value=0.0, format="%.2f")
        duration = st.number_input('Enter the duration (in minutes)', min_value=0.0, format="%.2f")

        # Allow the user to select traffic condition
        traffic_condition = st.radio(
            "Select Traffic Condition",
            ('Light Traffic', 'Moderate Traffic', 'Heavy Traffic'),
            index=estimate_traffic_condition(map_time_of_day(datetime.now().hour))
        )

        traffic_condition_mapping = {'Light Traffic': 0, 'Moderate Traffic': 1, 'Heavy Traffic': 2}
        traffic_condition = traffic_condition_mapping[traffic_condition]

        st.write(f"Time of Day: {map_time_of_day(datetime.now().hour)}")
        st.write(f"Selected Traffic Condition: {['Light', 'Moderate', 'Heavy'][traffic_condition]}")

        if distance > 0 and duration > 0:
            st.write(f"Entered Distance: {distance:.2f} km")
            st.write(f"Entered Duration: {duration:.2f} mins")

            if st.button('Predict Fare for Pincode to Pincode'):
                fare = predict_fare(best_rf_model, scaler, distance, duration, time_mapping[map_time_of_day(datetime.now().hour)], traffic_condition)
                st.write(f"Predicted Fare from Pincode {from_pincode} to Pincode {to_pincode}: {fare:.2f} INR")
else:
    st.write("One of the datasets is missing the required 'pin_code' or 'pincode' column.")
