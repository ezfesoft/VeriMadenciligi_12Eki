# -*- coding: utf-8 -*-
"""
================================================================================
DERS: Veri Madenciliği ve Makine Öğrenmesi Algoritmaları
HAFTA 4: "Model İnternette Değil, RAM'de!" - Model Felsefesi ve İlk Tahmin
KAYNAK KOD: hafta4_model_deneyimi.py
================================================================================
Bu script önlisans öğrencilerinin 'model' kavramını somutlaştırması için
hazırlanmıştır.
1. Yerel CSV dosyasını üretir ve Pandas ile okur.
2. X (Özellikler) ve y (Hedef) ayrımını gösterir.
3. 3 farklı modeli (k-NN, Karar Ağacı, Lojistik Regresyon) yerel RAM'de eğitir.
4. Hiç görülmemiş yeni karakterleri anında tahmin eder.
5. İnternet bağlantısı gerektirmediğini mikro-saniye bazlı süre ölçümüyle kanıtlar.
================================================================================
"""

import os
import sys
import time
import pandas as pd
import numpy as np

# scikit-learn kütüphanesinin kurulu olup olmadığını kontrol ediyoruz
try:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import NotFittedError
except ImportError:
    print("[HATA] 'scikit-learn' kütüphanesi bulunamadı!")
    print("Terminal veya Komut Satırında şu komutu çalıştırınız: pip install scikit-learn pandas numpy")
    sys.exit(1)


def csv_olustur_ve_kaydet(dosya_adi="hafta4_oyun_karakterleri.csv"):
    """
    Öğrencilerin dışarıdan veri seti aramakla vakit kaybetmemesi için
    oyun karakterleri tablosunu üretir ve yerel diske CSV olarak kaydeder.
    """
    veri = {
        'Karakter_Adi': [
            'Barbar', 'Gladyator', 'Paladin', 
            'Ates_Buyucusu', 'Buz_Cadisi', 'Nekromansir', 
            'Orman_Okcusu', 'Keskin_Nisanci', 'Golge_Avcisi', 
            'Kaya_Devi', 'Celik_Muhafiz', 'Demir_Sovalye'
        ],
        'Can_Puani': [2400, 2100, 2600, 900, 1000, 1100, 1200, 1100, 1300, 3500, 3200, 3000],
        'Saldiri_Gucu': [180, 200, 140, 260, 220, 210, 230, 270, 210, 90, 110, 120],
        'Menzil': [1, 1, 1, 8, 7, 6, 9, 10, 8, 1, 1, 1],
        'Zirh': [85, 75, 95, 20, 25, 30, 35, 25, 40, 120, 110, 105],
        'Sinif': [
            'Savasci', 'Savasci', 'Savasci', 
            'Buyucu', 'Buyucu', 'Buyucu', 
            'Okcu', 'Okcu', 'Okcu', 
            'Tank', 'Tank', 'Tank'
        ]
    }
    
    df = pd.DataFrame(veri)
    df.to_csv(dosya_adi, index=False, encoding='utf-8')
    print(f"\n[ADIM 1] '{dosya_adi}' dosyası yerel diske başarıyla kaydedildi.")
    return dosya_adi


def veriyi_hazirla(dosya_adi="hafta4_oyun_karakterleri.csv"):
    """
    CSV dosyasını okur; modeli beslemek için X (ipuçları) ve y (etiketler)
    şeklinde ikiye böler.
    """
    print("\n" + "="*70)
    print("ADIM 2: CSV DOSYASININ OKUNMASI VE X, y AYRIMI")
    print("="*70)
    
    # CSV'yi Pandas ile içe aktar
    df = pd.read_csv(dosya_adi)
    print(f"Toplam Yüklenen Satır Sayısı: {len(df)}")
    print("\n--- Veri Setinin İlk 4 Satırı ---")
    print(df.head(4).to_string(index=False))
    
    # Modelin bakacağı sayısal sütunlar (Özellikler matrisi: X)
    ozellik_kolonlari = ['Can_Puani', 'Saldiri_Gucu', 'Menzil', 'Zirh']
    X = df[ozellik_kolonlari]
    
    # Tahmin etmek istediğimiz hedef sütun (Hedef vektörü: y)
    y = df['Sinif']
    
    print("\n[TEORİK NOT]")
    print(" -> Büyük X : Modele sunduğumuz ipuçları tablosudur (2 Boyutlu Matris).")
    print(" -> Küçük y : Modelin tahmin etmesini istediğimiz doğru cevaptır (1 Boyutlu Seri).")
    
    return X, y, ozellik_kolonlari


def modelleri_tanimla():
    """
    3 Farklı algoritma nesnesini oluşturur.
    Burada modeller henüz hiçbir veri görmemiştir ('Boş Kutu' aşaması).
    """
    print("\n" + "="*70)
    print("ADIM 3: MODELLERİN NESNE OLARAK RAM'E YÜKLENMESİ (BOŞ KUTULAR)")
    print("="*70)
    
    model_knn = KNeighborsClassifier(n_neighbors=3)
    model_tree = DecisionTreeClassifier(random_state=42)
    model_logreg = LogisticRegression(max_iter=1000)
    
    print("1. KNeighborsClassifier (k-NN) oluşturuldu  -> Analoji: 'Bana arkadaşını söyle...'")
    print("2. DecisionTreeClassifier (Ağaç) oluşturuldu -> Analoji: '20 Soru / Akış Şeması'")
    print("3. LogisticRegression (Regresyon) oluşturuldu -> Analoji: 'Ağırlıklı Puan Terazisi'")
    
    return model_knn, model_tree, model_logreg


def modelleri_egit(X, y, model_knn, model_tree, model_logreg):
    """
    .fit(X, y) çağrılarak modeller eğitilir.
    Eğitimin yerel bellekte gerçekleştiğini kanıtlamak için süre mikro-saniye
    seviyesinde ölçülür.
    """
    print("\n" + "="*70)
    print("ADIM 4: MODELLERİN EĞİTİLMESİ (.fit) VE SÜRE ÖLÇÜMÜ")
    print("="*70)
    
    baslangic = time.perf_counter()
    
    # 3 Modeli de eğitiyoruz
    model_knn.fit(X, y)
    model_tree.fit(X, y)
    model_logreg.fit(X, y)
    
    bitis = time.perf_counter()
    gecen_sure_ms = (bitis - baslangic) * 1000
    
    print(f"[BAŞARILI] 3 Model de yerel RAM belleğinde eğitildi!")
    print(f"Toplam Eğitim Süresi: {gecen_sure_ms:.2f} milisaniye!")
    print("\n>>> DİKKAT: Hiçbir sunucuya bağlanılmadı, internet trafiği oluşmadı.")
    print(">>> Model artık Python'ın RAM'inde bir değişken (nesne) olarak yaşıyor.")


def yeni_verilerle_test_et(model_knn, model_tree, model_logreg, ozellik_kolonlari):
    """
    Modelin eğitimde HİÇ GÖRMEDİĞİ 2 yeni karakteri test eder.
    .predict() fonksiyonunun çalışmasını gösterir.
    """
    print("\n" + "="*70)
    print("ADIM 5: HİÇ GÖRÜLMEMİŞ YENİ KARAKTERLERLE İLK TAHMİN (.predict)")
    print("="*70)
    
    # 1. Yeni Karakter: Can: 3300, Saldırı: 105, Menzil: 1, Zırh: 115 (Tipik Tank profili)
    # 2. Yeni Karakter: Can: 920,  Saldırı: 255, Menzil: 8, Zırh: 22  (Tipik Büyücü profili)
    yeni_test_verisi = pd.DataFrame([
        [3300, 105, 1, 115],
        [920, 255, 8, 22]
    ], columns=ozellik_kolonlari)
    
    print("Modele Sunulan Yeni Karakterler:")
    print(yeni_test_verisi)
    
    # Tahminleri toplayalım
    tahminler_knn = model_knn.predict(yeni_test_verisi)
    tahminler_tree = model_tree.predict(yeni_test_verisi)
    tahminler_logreg = model_logreg.predict(yeni_test_verisi)
    
    karsilastirma_df = pd.DataFrame({
        'Karakter': ['Bilinmeyen Aday 1', 'Bilinmeyen Aday 2'],
        'İnsan Sezgisi': ['Tank Olmalı', 'Büyücü/Okçu Olmalı'],
        'k-NN Tahmini': tahminler_knn,
        'Karar Ağacı': tahminler_tree,
        'Lojistik Reg.': tahminler_logreg
    })
    
    print("\n--- MODELLERİN KARŞILAŞTIRMALI TAHMİN TABLOSU ---")
    print(karsilastirma_df.to_string(index=False))


def egitilmemis_model_deneyi(ozellik_kolonlari):
    """
    Öğrencilere .fit() yapmadan .predict() çağrılırsa ne olacağını gösterir.
    Bu deney modelin sihir değil, durum tutan bir algoritma olduğunu kanıtlar.
    """
    print("\n" + "="*70)
    print("ADIM 6: PEDAGOJİK DENEY — FIT EDİLMEMİŞ MODEL NE YAPAR?")
    print("="*70)
    
    ham_model = DecisionTreeClassifier()
    ornek = pd.DataFrame([[2000, 150, 1, 80]], columns=ozellik_kolonlari)
    
    try:
        ham_model.predict(ornek)
    except NotFittedError as e:
        print("[BEKLENEN HATA YAKALANDI - NotFittedError]:")
        print(f"Mesaj: '{e}'")
        print("\nÇIKARILACAK DERS: Model eğitilmeden (.fit edilmeden) hiçbir şey bilemez!")
        print("Çünkü henüz RAM'inde karar verecek kurallar veya ağırlıklar yoktur.")


def ogrenci_karakter_olusturucu(model_knn, model_tree, model_logreg, ozellik_kolonlari):
    """
    Öğrencinin veya hocanın terminal üzerinden kendi uydurduğu değerleri
    girip anında 3 modelden de canlı tahmin almasını sağlar.
    """
    print("\n" + "="*70)
    print("ADIM 7: CANLI ETKİLEŞİM — KENDİ KARAKTERİNİ YARAT")
    print("="*70)
    print("Örnek değerler: Can: 1000-3500 | Saldırı: 80-280 | Menzil: 1-10 | Zırh: 20-120")
    
    ornek_degerler = [
        ("Örnek Suikastçı Tarzı", 1400, 220, 2, 45),
        ("Örnek Süper Tank", 3600, 80, 1, 130),
    ]
    
    for aciklama, can, saldiri, menzil, zirh in ornek_degerler:
        kullanici_verisi = pd.DataFrame([[can, saldiri, menzil, zirh]], columns=ozellik_kolonlari)
        print(f"\n>> Test: {aciklama} (Can:{can}, Saldırı:{saldiri}, Menzil:{menzil}, Zırh:{zirh})")
        print(f"   -> k-NN Kararı        : {model_knn.predict(kullanici_verisi)[0]}")
        print(f"   -> Karar Ağacı Kararı : {model_tree.predict(kullanici_verisi)[0]}")
        print(f"   -> Lojistik Regresyon : {model_logreg.predict(kullanici_verisi)[0]}")


def main():
    print("*" * 75)
    print("  HAFTA 4: MAKİNE ÖĞRENMESİ VE MODEL FELSEFESİ UYGULAMASI BAŞLATILIYOR")
    print("*" * 75)
    
    # 1. CSV üretimi
    csv_dosyasi = csv_olustur_ve_kaydet()
    
    # 2. Veri hazırlığı (X ve y)
    X, y, ozellikler = veriyi_hazirla(csv_dosyasi)
    
    # 3. Model tanımlamaları
    knn, tree, logreg = modelleri_tanimla()
    
    # 4. Yerel RAM'de eğitim
    modelleri_egit(X, y, knn, tree, logreg)
    
    # 5. Görülmemiş verilerle tahmin
    yeni_verilerle_test_et(knn, tree, logreg, ozellikler)
    
    # 6. Eğitilmemiş model hatası deneyi
    egitilmemis_model_deneyi(ozellikler)
    
    # 7. Etkileşimli karakter denemeleri
    ogrenci_karakter_olusturucu(knn, tree, logreg, ozellikler)
    
    print("\n" + "*" * 75)
    print("  UYGULAMA TAMAMLANDI! ÖĞRENCİLER İÇİN TARTIŞMA SORUSU:")
    print("  'Wi-Fi'ı tamamen kapatsanız bu kodun çıktısı değişir mi?' -> CEVAP: ASLA!")
    print("*" * 75 + "\n")


if __name__ == "__main__":
    main()