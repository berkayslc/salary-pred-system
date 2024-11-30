
# Yazılım Geliştiricileri İçin Maaş Tahmini Uygulaması 

Bu proje, Python ve Streamlit kullanılarak geliştirilmiş bir maaş tahmin uygulamasıdır. Kullanıcı, pozisyon, deneyim seviyesi, lokasyon, çalışma şekli, çalışan sayısı ve deneyim süresine göre tahmini maaşı öğrenebilir.

## Özellikler

- **Model Kullanımı:** Daha önce eğitilmiş bir makine öğrenimi modeli (`salary_prediction_model.pkl`) kullanılarak maaş tahmini yapılır.
- **Label Encoding:** Kategorik veriler `label_encoders.pkl` dosyası kullanılarak sayısal değerlere dönüştürülür.
- **Dinamik Veri Seçimi:** Kullanıcı, uygulama arayüzünde mevcut pozisyon, seviye, lokasyon gibi seçeneklerden seçim yapabilir.
- **Tahmin Gösterimi:** Tahmini maaş, kullanıcıya anlaşılır ve formatlanmış bir şekilde gösterilir.

## Kullanılan Kütüphaneler

- **pandas:** Veri işleme ve manipülasyon.
- **pickle:** Eğitilmiş modeli ve label encoder'ları yükleme.
- **streamlit:** Kullanıcı dostu bir arayüz oluşturma.

## Nasıl Çalıştırılır?

1. Gerekli kütüphaneleri yükleyin:
    ```bash
    pip install pandas streamlit
    ```
2. Uygulamayı çalıştırmak için aşağıdaki komutu kullanın:
    ```bash
    streamlit run app.py
    ```
3. Tarayıcınızda `http://localhost:8501` adresine giderek uygulamayı kullanabilirsiniz.

## Girdi Verileri

Kullanıcının tahmin için sağlaması gereken veriler:
- **Pozisyon:** İş unvanı (ör. Backend Developer, Data Scientist).
- **Deneyim Seviyesi:** Junior, Mid, Senior gibi.
- **Lokasyon:** Türkiye veya diğer ülkeler.
- **Çalışma Şekli:** Uzaktan, Hibrit veya Ofiste.
- **Çalışan Sayısı:** Şirketin çalışan sayısı kategorisi.
- **Deneyim Süresi:** Yıllarla ifade edilen deneyim süresi.

## Çıktı

Tahmini maaş, TL cinsinden formatlanmış şekilde gösterilir:
```
Tahmini Maaş: **25,000.00 TL**
```

## Dosyalar

- `app.py`: Ana uygulama kodu.
- `salary_prediction_model.pkl`: Eğitilmiş maaş tahmin modeli.
- `label_encoders.pkl`: Kategorik verileri sayısal değerlere dönüştüren encoder'lar.
- `salary.csv`: Örnek veri seti.
