# Gerekli kütüphaneleri içe aktarıyoruz
import pandas as pd  # Veri işleme için
import pickle as pk  # Model ve label encoder dosyalarını yüklemek için
import streamlit as st  # Web tabanlı bir kullanıcı arayüzü oluşturmak için

# Daha önce eğitilmiş maaş tahmin modeli dosyasını yüklüyoruz
model = pk.load(open('KNeighborsRegressor.pkl', 'rb'))

# Label encoder'lar kategorik verileri modelin anlayabileceği şekilde dönüştürmek için kullanılır
label_encoders = pk.load(open('KNeighborsRegresso_label_encoder.pkl', 'rb'))

# Maaş tahmini için örnek veri setini yüklüyoruz (kullanıcıya seçenek sunabilmek için)
df = pd.read_csv('salary.csv')

# Streamlit ile uygulama başlığını belirtiyoruz
st.title("Yazılım Geliştiricileri İçin Maaş Tahmini Uygulaması")
st.write("Pozisyon, deneyim seviyesi ve çalışma koşullarına göre maaş tahmini yapabilirsiniz.")

# Kullanıcıya sunulacak seçenekleri (benzersiz değerleri) belirlemek için veri setindeki sütunları analiz ediyoruz
unique_positions = df['Position'].unique()  # Benzersiz iş unvanları
unique_levels = df['Level'].unique()  # Benzersiz deneyim seviyeleri
unique_locations = df['Location'].unique()  # Benzersiz lokasyonlar
unique_working_modes = df['Way_of_working'].unique()  # Benzersiz çalışma şekilleri
unique_employee_numbers = df['Employees_number'].unique()  # Benzersiz çalışan sayısı değerleri

# Kullanıcıdan uygulamada tahmin için gerekli verileri girmesini bekliyoruz
position = st.selectbox('Pozisyon', unique_positions)  # Kullanıcı pozisyon seçiyor
level = st.selectbox('Deneyim Seviyesi', unique_levels)  # Kullanıcı deneyim seviyesini seçiyor
location = st.selectbox('Lokasyon', unique_locations)  # Kullanıcı lokasyon seçiyor
way_of_working = st.selectbox('Çalışma Şekli', unique_working_modes)  # Kullanıcı çalışma şeklini seçiyor
employees_number = st.selectbox('Çalışan Sayısı', unique_employee_numbers)  # Kullanıcı çalışan sayısını seçiyor
experience = st.slider('Deneyim Süresi (Yıl)', 0, 20, 5)  # Kullanıcı deneyim süresini (yıllarla) seçiyor (min: 0, max: 20, varsayılan: 5 yıl)

# Kullanıcı tahmin butonuna bastığında işlem başlıyor
if st.button("Tahmin Et"):
    # Kullanıcının girdiği verileri model için uygun bir DataFrame formatına dönüştürüyoruz
    input_data = pd.DataFrame({
        'Position': [position],  # Seçilen pozisyon
        'Level': [level],  # Seçilen deneyim seviyesi
        'Location': [location],  # Seçilen lokasyon
        'Way_of_working': [way_of_working],  # Seçilen çalışma şekli
        'Employees_number': [employees_number],  # Seçilen çalışan sayısı kategorisi
        'Avg_Experience': [experience]  # Seçilen deneyim süresi
    })

    # Kategorik verileri label encoder ile sayısal değerlere dönüştürüyoruz
    # Model bu sayısal değerlerle tahmin yapar
    for col in ['Position', 'Level', 'Location', 'Way_of_working', 'Employees_number']:
        input_data[col] = label_encoders[col].transform(input_data[col])

    # Kullanıcıdan alınan verileri eğitilmiş modele vererek tahmin işlemini gerçekleştiriyoruz
    predicted_salary_f_2022 = model.predict(input_data)[0] # Modelin döndürdüğü ilk (ve tek) tahmin sonucu 
    predicted_salary_f_2024 = model.predict(input_data)[0]*2.06  # Modelin döndürdüğü ilk (ve tek) tahmin sonucu 2024 için 2.06 ile çarpılmıştır. Enflasyon hesaba katıldı.

    # Kullanıcıya tahmini maaşı gösteriyoruz
    st.write(f"2023 Tahmini Maaş: **{predicted_salary_f_2022:,.2f} TL**")  # Maaşı düzgün formatta (ör. 1.000,00 TL) gösteriyoruz
    st.write(f"2024 Tahmini Maaş: **{predicted_salary_f_2024:,.2f} TL**")  # Maaşı düzgün formatta (ör. 1.000,00 TL) gösteriyoruz
