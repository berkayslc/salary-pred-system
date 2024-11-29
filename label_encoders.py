import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Veri setini yükleyin
df = pd.read_csv('salary.csv')

# Dönüştürülecek kategorik sütunlar
categorical_columns = ['Position', 'Level', 'Location', 'Way_of_working', 'Employees_number']

# Label Encoders oluştur ve her sütun için eğit
label_encoders = {col: LabelEncoder() for col in categorical_columns}

for col in categorical_columns:
    df[col] = label_encoders[col].fit_transform(df[col])

# Label encoders'ı bir dosyaya kaydedin
with open('label_encoders.pkl', 'wb') as file:
    pickle.dump(label_encoders, file)

print("label_encoders.pkl dosyası oluşturuldu.")
