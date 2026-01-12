# CarDetectionBilling 
Real-time vehicle detection &amp; pricing

# 🚗 Car Detection & Billing System

YouTube videoları üzerinden araçları tespit eden, takip eden ve geçiş ücreti hesaplayan akıllı bir sistem.

## 🎯 Amaç

- Araçları video akışında tespit etmek  
- Aynı aracı tekrar tanıyarak takip etmek  
- İlk geçişte araç sınıfına göre ücret almak  
- Toplam araç sayısını ve geliri canlı olarak göstermek  

Bu sistem, otopark yönetimi, köprü geçişleri veya ücretli yol sistemleri için temel bir çözüm sunar.

## 🧠 Kullanılan Teknolojiler

- **Python** – Ana programlama dili  
- **YOLOv8n (nano)** – Nesne tespiti için  
- **SORT** – Takip algoritması  
- **OpenCV** – Görüntü işleme  
- **yt-dlp** – YouTube videosunu çekmek için  

## ⚙️ Özellikler

- [x] `car`, `bus`, `truck`, `motorcycle`, `bicycle` sınıflarının tespiti  
- [x] Araçların tekrar geçişini algılama (aynı araca bir kez ücret uygulanır)  
- [x] Her araç tipi için özel ücretlendirme  
- [x] Üst panelde toplam araç sayısı ve gelir gösterimi  
- [x] "q" tuşuyla uygulamayı sonlandırma  

## 💰 Ücretlendirme (Örnek)

| Araç Tipi     | Ücret (₺) |
|---------------|-----------|
| Car           | 20        |
| Bus           | 30        |
| Truck         | 40        |
| Motorcycle    | 10        |
| Bicycle       | 5         |

> Ücret değerleri `main.py` içerisinde kolayca düzenlenebilir.

## 🚀 Kurulum

```bash
git clone https://github.com/YusufTufan/CarDetectionBilling.git
cd CarDetectionBilling
pip install -r requirements.txt
````

> **Not:** YOLOv8 modeli için `ultralytics` kütüphanesini de yüklemeniz gerekebilir:

```bash
pip install ultralytics
```

## ▶️ Kullanım

1. `main.py` dosyasını çalıştırın.
2. YouTube videosu otomatik olarak çekilir ve analiz başlar.
3. Araçlar ekranda kutularla gösterilir, sınıflar etiketlenir.
4. Üst panelde gerçek zamanlı toplam gelir ve araç sayısı yer alır.
5. `"q"` tuşuna basıldığında uygulama güvenli şekilde kapanır.


---

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

Copyright (c) 2025 YusufTufan
