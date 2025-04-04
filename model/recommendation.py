import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def load_dataset(file_path):
    data = pd.read_csv(file_path)
    print(f"Dataset loaded with shape: {data.shape}")
    return data

data_1_path = 'data/Updated_FoodRecommandation_With_Prices (1).csv'
data_2_path = 'data/Large_Generated_Orders_Dataset (1) (1).csv'

data_1 = load_dataset(data_1_path)
data_2 = load_dataset(data_2_path)

def preprocess_data(data):
    data['Preparation Time (Minutes)'] = pd.to_numeric(data['Preparation Time (Minutes)'], errors='coerce')
    data['Ingredient Count'] = pd.to_numeric(data['Ingredient Count'], errors='coerce')
    data['Selling Price (EGP)'] = pd.to_numeric(data['Selling Price (EGP)'], errors='coerce')
    data['Cost Price (EGP)'] = pd.to_numeric(data['Cost Price (EGP)'], errors='coerce')
    return data.dropna()

data_1 = preprocess_data(data_1)

scaler = StandardScaler()

def recommend_food(prep_time, num_ingredients, country, selling_price, cost_price, n_recommendations=5):
    data_encoded = pd.get_dummies(data_1, columns=['Country'], drop_first=True)
    features = ['Preparation Time (Minutes)', 'Ingredient Count', 'Rating', 'Selling Price (EGP)', 'Cost Price (EGP)'] + \
               [col for col in data_encoded.columns if col.startswith('Country_')]
    
    X = data_encoded[features]
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    model = NearestNeighbors(n_neighbors=n_recommendations, metric='euclidean')
    model.fit(X_scaled)

    country_col = f"Country_{country}"
    input_vector = np.zeros(len(features))
    if country_col in features:
        input_vector[features.index(country_col)] = 1
    input_vector[features.index('Preparation Time (Minutes)')] = prep_time
    input_vector[features.index('Ingredient Count')] = num_ingredients
    input_vector[features.index('Selling Price (EGP)')] = selling_price
    input_vector[features.index('Cost Price (EGP)')] = cost_price
    input_vector_scaled = scaler.transform([input_vector])

    if country in data_1['Country'].unique():
        filtered_data = data_1[data_1['Country'] == country]
    else:
        print(f"No data available for country: {country}. Recommending based on other factors.")
        filtered_data = data_1

    filtered_data_encoded = pd.get_dummies(filtered_data, columns=['Country'], drop_first=True)
    filtered_data_encoded = filtered_data_encoded.reindex(columns=features, fill_value=0) 
    filtered_X_scaled = scaler.transform(filtered_data_encoded[features])

    model.fit(filtered_X_scaled)

    distances, indices = model.kneighbors(input_vector_scaled)
    recommendations = filtered_data.iloc[indices[0]].to_dict(orient='records')
    return recommendations

def recommend_based_on_country_and_profit(selected_country, top_n=5):
    data_1['Profit'] = data_1['Selling Price (EGP)'] - data_1['Cost Price (EGP)']

    country_data = data_1[data_1['Country'] == selected_country]

    if not country_data.empty:
        sorted_data = country_data.sort_values(by='Profit', ascending=False).head(top_n)
        return sorted_data.to_dict(orient='records')
    else:
        print(f"No data available for country: {selected_country}. Recommending from other countries.")
        
        feature_columns = ['Preparation Time (Minutes)', 'Ingredient Count', 'Rating', 'Selling Price (EGP)', 'Cost Price (EGP)', 'Profit']
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_1[feature_columns])
        
        kmeans = KMeans(n_clusters=5, random_state=42)
        data_1['Cluster'] = kmeans.fit_predict(scaled_data)
        
        recommendations = []
        for cluster in data_1['Cluster'].unique():
            cluster_data = data_1[data_1['Cluster'] == cluster]
            top_item = cluster_data.sort_values(by='Profit', ascending=False).head(1)
            recommendations.append(top_item)
        
        recommendations_df = pd.concat(recommendations).sort_values(by='Profit', ascending=False).head(top_n)
        return recommendations_df.to_dict(orient='records')

def collaborative_recommendation_with_similarity(country, n_recommendations=5):
    # data_1 = pd.read_csv('data_1.csv')
    # data_2 = pd.read_csv('data_2.csv')

    data_1['Name'] = data_1['Name'].str.lower().str.strip()
    data_2['Item Name'] = data_2['Item Name'].str.lower().str.strip()

    merged_data = pd.merge(data_2, data_1, left_on='Item Name', right_on='Name', how='inner')

    merged_data['Profit'] = merged_data['Selling Price (EGP)_y'] - merged_data['Cost Price (EGP)']
    feature_columns = ['Preparation Time (Minutes)', 'Ingredient Count', 'Rating', 'Profit']
    scaler = StandardScaler()
    merged_data[feature_columns] = scaler.fit_transform(merged_data[feature_columns])

    item_features = merged_data[feature_columns].values
    similarity_matrix = cosine_similarity(item_features)

    merged_data['Similarity Score'] = similarity_matrix.sum(axis=1)

    if country in merged_data['Country'].unique():
        country_data = merged_data[merged_data['Country'] == country]
        recommendations = country_data.sort_values(
            by=['Similarity Score', 'Profit', 'Rating'],
            ascending=[False, False, False]
        ).drop_duplicates(subset=['Item Name'])
    else:
        print(f"No data available for country: {country}. Recommending based on other factors.")
        recommendations = merged_data.sort_values(
            by=['Similarity Score', 'Profit', 'Rating'],
            ascending=[False, False, False]
        ).drop_duplicates(subset=['Item Name'])

    recommended_names = recommendations['Item Name'].head(n_recommendations).tolist()
    final_recommendations = data_1[data_1['Name'].isin(recommended_names)]

    return final_recommendations.sort_values(by='Rating', ascending=False).to_dict(orient='records')
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_error

def evaluate_model():
    total_mean_distance = 0
    max_distance = 0
    accurate_predictions = 0
    total_predictions = len(data_1)

    # استخراج الخصائص لاستخدامها في الـ KMeans و NearestNeighbors
    feature_columns = ['Preparation Time (Minutes)', 'Ingredient Count', 'Rating', 'Selling Price (EGP)', 'Cost Price (EGP)']
    X = data_1[feature_columns]
    
    # تطبيق التقييس
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # تطبيق KMeans للتصنيف
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # تقسيم البيانات إلى مجموعة تدريب (train) واختبار (test)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, clusters, test_size=0.2, random_state=42)

    # تدريب نموذج NearestNeighbors على مجموعة التدريب
    model = NearestNeighbors(n_neighbors=5, metric='euclidean')
    model.fit(X_train)

    # تقييم النموذج عبر الاختبار
    for i in range(len(X_test)):
        distances, indices = model.kneighbors([X_test[i]], n_neighbors=5)
        total_mean_distance += distances.mean()
        max_distance = max(max_distance, distances.max())

        # حساب الدقة - التنبؤ بالمجموعة الصحيحة
        true_cluster = y_test[i]
        predicted_clusters = [y_train[idx] for idx in indices[0]]
        
        # إذا كانت المجموعة الصحيحة موجودة ضمن أقرب الجيران
        if true_cluster in predicted_clusters:
            accurate_predictions += 1

    # حساب المتوسط للدقة والمسافات
    avg_mean_distance = total_mean_distance / len(X_test)
    accuracy = accurate_predictions / len(X_test)

    # طباعة النتائج
    print("=== Model Evaluation Results ===")
    print("Average Mean Distance (lower is better):", avg_mean_distance)
    print("Max Distance (outlier indicator):", max_distance)
    print("Accuracy (higher is better):", accuracy)

    return avg_mean_distance, accuracy

# item_to_recommend = "hamburger 119"
# recommendations_food = recommend_food(30, 5, "Italy", 150, 100, n_recommendations=3)
# recommendations_profit = recommend_based_on_country_and_profit("Italy", top_n=3)
# recommendations_merge = collaborative_recommendation_with_similarity(data_1, data_2,"ُItaly",n_recommendations=3)

# print("=== Recommendations Based on Food Properties ===")
# for idx, dish in enumerate(recommendations_food, 1):
#     print(f"""{idx}. Dish: {dish['Name']}
#         -Category: {dish['Category']}
#         - Country: {dish['Country']}
#         -Selling Price: {dish['Selling Price (EGP)']}
#         -Cost Price: {dish['Cost Price (EGP)']}
#         - Preparation Method: {dish['Preparation Method']}
#         - Ingredient Count: {dish['Ingredient Count']}
#         -Ingredients: {dish['Ingredients']}
#         - Preparation Time: {dish['Preparation Time (Minutes)']} minutes
#         """)
# print("\n=== Recommendations Based on Profit by Country ===")
# for idx, dish in enumerate(recommendations_profit, 1):
#     print(f"""{idx}. Dish: {dish['Name']}
#         -Category: {dish['Category']}
#         - Country: {dish['Country']}
#         -Selling Price: {dish['Selling Price (EGP)']}
#         -Cost Price: {dish['Cost Price (EGP)']}
#         - Preparation Method: {dish['Preparation Method']}
#         - Ingredient Count: {dish['Ingredient Count']}
#         -Ingredients: {dish['Ingredients']}
#         - Preparation Time: {dish['Preparation Time (Minutes)']} minutes
#         """)
# print("\n=== Recommendations Based on Merged Datasets ===")
# for idx, dish in enumerate(recommendations_merge, 1):
#     print(f"""{idx}. Dish: {dish['Name']}
#         -Category: {dish.get('Category', 'N/A')}
#         -Country: {dish.get('Country', 'N/A')}
#         -Selling Price: {dish.get('Selling Price (EGP)', 'N/A')}
#         -Cost Price: {dish.get('Cost Price (EGP)', 'N/A')}
#         -Rating: {dish.get('Rating', 'N/A')}
#         -Ingredients: {dish.get('Ingredients', 'N/A')}
#         -Preparation Time: {dish.get('Preparation Time (Minutes)', 'N/A')} minutes
#     """)


# print("\n=== Evalute Model ===")


# evaluate_model()