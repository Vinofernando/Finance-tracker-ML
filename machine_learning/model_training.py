from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

# 1. PERBARUI DATA LATIH SESUAI KATEGORI ASLI DI DB-MU
data_deskripsi = [
    "Isi bensin pertamax di pertamina",   # Transportasi
    "Bayar tarif tol dan parkir emoney", # Transportasi
    "Ganti oli motor beat di bengkel",   # Transportasi
    
    "Beli nasi padang siang hari",       # Makanan/minuman
    "Kopi susu janji jiwa senja",        # Makanan/minuman
    "Makan malam sate ayam madura",      # Makanan/minuman
    
    "Transfer ke rekening bank mandiri", # Pemindahan dana
    "Kirim uang ke e-wallet gopay",      # Pemindahan dana
    "Top up ovo lewat blu bca",          # Pemindahan dana
    
    "Gaji bulanan dari kantor masuk",    # gaji
    "Bonus project fullstack web",       # gaji
    "Transferan fee freelance",          # gaji
    "gaji",
    
    "Beli kasur busa baru",              # lainnya
    "Biaya admin bulanan bank",          # lainnya
    "Nonton di 21 bcs"
]

# Pastikan teks kategori di sini SAMA PERSIS (termasuk huruf besar-kecilnya) dengan di database-mu
data_kategori = [
    1, 1, 1,
    2, 2, 2,
    3, 3, 3,
    4, 4, 4, 4,
    5, 5, 5
]

data_type = [
    "expense", "expense", "expense",
    "expense", "expense", "expense",
    "expense", "expense", "expense",
    "income", "income", "income", "income",
    "expense", "expense", "expense"
]

print("Melatih otak kategori")
model_kategori = make_pipeline(CountVectorizer(), MultinomialNB())
model_kategori.fit(data_deskripsi, data_kategori)

print("Melatih otak type")
model_type = make_pipeline(CountVectorizer(), MultinomialNB())
model_type.fit(data_deskripsi, data_type)

joblib.dump(model_kategori, 'model_kategori.pkl')
joblib.dump(model_type, 'model_type.pkl')
print("Kedua otak ML berhasil dibuat dan disimpan!\n")

tes_input = ["makan soto ayam", "gaji ke-13 masuk rejeki", "transfer dana ke emak", "Nonton di 21 bcs"]
prediksi_kategori = model_kategori.predict(tes_input)
prediksi_type = model_type.predict(tes_input)

for deskripsi, kategori, type in zip(tes_input, prediksi_kategori, prediksi_type):
    print(f"'{deskripsi}' ➔ Kategori: [{kategori}] | Type: [{type}]")
