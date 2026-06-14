from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
import joblib
import os
from django.conf import settings
from rest_framework import status

# Load model sekali saja saat server start
MODEL_PATH_CATEGORY = os.path.join(settings.BASE_DIR, 'model_kategori.pkl')
MODEL_PATH_TYPE = os.path.join(settings.BASE_DIR, 'model_type.pkl')
model_category = joblib.load(MODEL_PATH_CATEGORY)
model_type = joblib.load(MODEL_PATH_TYPE)

@api_view(['POST'])
@permission_classes([AllowAny]) # Supaya bisa ditembak langsung oleh backend utamamu
def prediksi_kategori_finance(request):
    try:
        deskripsi_transaksi = request.data.get('deskripsi')

        if not deskripsi_transaksi:
            return Response({"error": "Field deskripsi wajib diisi"}, status=status.HTTP_400_BAD_REQUEST)
        
        hasil_prediksi_kategori = model_category.predict([deskripsi_transaksi])
        hasil_prediksi_type = model_type.predict([deskripsi_transaksi])

        kategori_terpilih = hasil_prediksi_kategori[0]
        type_terpilih = hasil_prediksi_type[0]

        return Response({
            "status": "Success",
            "deskripsi": deskripsi_transaksi,
            "kategori_otomatis": kategori_terpilih,
            "type_otomatis": type_terpilih
        }, status=status.HTTP_200_OK)
    
    except Exception as e:
        return Response({"status": "Error", "message": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)