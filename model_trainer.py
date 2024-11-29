import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

# Veri setini yükleme
df = pd.read_csv('salary.csv')

# Maaş sütununu temizleme ve hesaplama
df['Salary'] = df['Salary'].str.replace('+', '', regex=False).str.replace(' TL', '', regex=False)
df['Min_Salary'] = df['Salary'].apply(lambda x: int(x.split('-')[0].replace('.', '')) if '-' in x else int(x.replace('.', '')))
df['Max_Salary'] = df['Salary'].apply(lambda x: int(x.split('-')[1].replace('.', '')) if '-' in x else int(x.replace('.', '')))
df['Avg_Salary'] = (df['Min_Salary'] + df['Max_Salary']) / 2

# Deneyim süresi (Experience) sütununu işleme
df['Min_Experience'] = df['Experience'].apply(lambda x: int(x.split('-')[0].replace('+', '').replace(' Yıl', '').strip()))
df['Max_Experience'] = df['Experience'].apply(lambda x: int(x.split('-')[1].replace('+', '').replace(' Yıl', '').strip()) if '-' in x else int(x.replace('+', '').replace(' Yıl', '').strip()))
df['Avg_Experience'] = (df['Min_Experience'] + df['Max_Experience']) / 2

# Kategorik değişkenleri sayısal değerlere dönüştürme
categorical_columns = ['Position', 'Level', 'Location', 'Way_of_working', 'Employees_number']
label_encoders = {col: LabelEncoder() for col in categorical_columns}

for col in categorical_columns:
    df[col] = label_encoders[col].fit_transform(df[col])

# Bağımsız ve bağımlı değişkenleri ayırma
X = df[['Position', 'Level', 'Location', 'Way_of_working', 'Employees_number', 'Avg_Experience']]
y = df['Avg_Salary']

# Veri setini eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model oluşturma ve eğitme
model = KNeighborsRegressor()
model.fit(X_train, y_train)

# Tahminler
y_pred = model.predict(X_test)

# Performans değerlendirme
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}")

# Modeli kaydetme
with open('KNeighborsRegressor.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Label encoders'ı kaydetme
with open('KNeighborsRegresso_label_encoder.pkl', 'wb') as encoders_file:
    pickle.dump(label_encoders, encoders_file)

print("Model ve label encoders başarıyla kaydedildi.")
