from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from typing import Optional
import json

# إنشاء تطبيق FastAPI
app = FastAPI(
    title="Life Expectancy Prediction API",
    description="API للتنبؤ بمتوقع العمر بناءً على المؤشرات الصحية والاقتصادية",
    version="1.0.0"
)

# تحميل النموذج والبيانات المحفوظة
try:
    model = joblib.load('life_expectancy_model.joblib')
    selected_features = joblib.load('selected_features.joblib')
    encoders = joblib.load('encoders.joblib')
    
    # تحميل البيانات الأصلية لمعرفة قيم الدول
    original_data = pd.read_csv('deathrate.csv')
    original_data.columns = original_data.columns.str.strip()
    
    # إنشاء قاموس ترميز الدول
    with open('target_column.txt', 'r', encoding='utf-8') as f:
        target_column = f.read().strip()
    
    country_encoding = original_data.groupby('Country')[target_column].mean().to_dict()
    available_countries = list(country_encoding.keys())
    
    print("✅ تم تحميل النموذج والمحولات بنجاح")
    print(f"المتغيرات المطلوبة: {selected_features}")
    
except Exception as e:
    print(f"❌ خطأ في تحميل النموذج: {e}")
    model = None
    selected_features = None
    encoders = None
    country_encoding = None
    available_countries = []

# نموذج البيانات المدخلة
class PredictionInput(BaseModel):
    Country: str = Field(..., description="اسم الدولة")
    Year: int = Field(..., description="السنة", ge=1990, le=2030)
    Adult_Mortality: float = Field(..., description="معدل الوفيات للبالغين (لكل 1000)", ge=0)
    infant_deaths: int = Field(..., description="وفيات الأطفال الرضع", ge=0)
    percentage_expenditure: float = Field(..., description="نسبة الإنفاق على الصحة من GDP", ge=0)
    under_five_deaths: int = Field(..., description="وفيات الأطفال تحت 5 سنوات", ge=0) 
    Diphtheria: float = Field(..., description="نسبة التطعيم ضد الدفتيريا (%)", ge=0, le=100)
    HIV_AIDS: float = Field(..., description="معدل الوفيات بسبب HIV/AIDS (لكل 1000)", ge=0)
    Schooling: float = Field(..., description="متوسط سنوات التعليم", ge=0, le=25)
    
    class Config:
        schema_extra = {
            "example": {
                "Country": "Jordan",
                "Year": 2015,
                "Adult_Mortality": 150,
                "infant_deaths": 5,
                "percentage_expenditure": 500.0,
                "under_five_deaths": 6,
                "Diphtheria": 95.0,
                "HIV_AIDS": 0.1,
                "Schooling": 13.5
            }
        }

# نموذج الاستجابة
class PredictionOutput(BaseModel):
    predicted_life_expectancy: float = Field(..., description="متوقع العمر المتوقع (بالسنوات)")
    confidence: str = Field(..., description="مستوى الثقة في التنبؤ")
    message: str = Field(..., description="رسالة توضيحية")

# التحقق من صحة البيانات المدخلة
def validate_and_prepare_input(input_data: PredictionInput):
    """التحقق من صحة البيانات وتحضيرها للنموذج"""
    
    # التحقق من وجود الدولة
    if input_data.Country not in available_countries:
        # البحث عن أقرب دولة
        similar_countries = [c for c in available_countries if input_data.Country.lower() in c.lower() or c.lower() in input_data.Country.lower()]
        if similar_countries:
            suggestion = similar_countries[0]
            raise HTTPException(
                status_code=400, 
                detail=f"الدولة '{input_data.Country}' غير موجودة. هل تقصد: {suggestion}؟\nللحصول على قائمة الدول المتاحة، استخدم /countries"
            )
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"الدولة '{input_data.Country}' غير موجودة في البيانات.\nللحصول على قائمة الدول المتاحة، استخدم /countries"
            )
    
    # تحضير البيانات للنموذج
    data_dict = {}
    
    # المتغيرات المطلوبة بالترتيب الصحيح
    feature_mapping = {
        'Year': input_data.Year,
        'Adult Mortality': input_data.Adult_Mortality,
        'infant deaths': input_data.infant_deaths,
        'percentage expenditure': input_data.percentage_expenditure,
        'under-five deaths': input_data.under_five_deaths,
        'Diphtheria': input_data.Diphtheria,
        'HIV/AIDS': input_data.HIV_AIDS,
        'Schooling': input_data.Schooling,
        'Country_encoded': country_encoding[input_data.Country]
    }
    
    # التأكد من وجود جميع المتغيرات المطلوبة
    for feature in selected_features:
        if feature in feature_mapping:
            data_dict[feature] = feature_mapping[feature]
        else:
            raise HTTPException(
                status_code=500,
                detail=f"المتغير {feature} مفقود من البيانات المدخلة"
            )
    
    return pd.DataFrame([data_dict])

@app.get("/")
async def root():
    """الصفحة الرئيسية للAPI"""
    return {
        "message": "مرحباً بك في API التنبؤ بمتوقع العمر",
        "version": "1.0.0",
        "model_accuracy": "96.3%",
        "endpoints": {
            "/predict": "التنبؤ بمتوقع العمر",
            "/health": "فحص حالة API", 
            "/countries": "قائمة الدول المتاحة",
            "/docs": "الوثائق التفاعلية"
        }
    }

@app.get("/health")
async def health_check():
    """فحص حالة API"""
    if model is None:
        return {"status": "error", "message": "النموذج غير محمل"}
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "features_count": len(selected_features) if selected_features else 0,
        "countries_available": len(available_countries)
    }

@app.get("/countries")
async def get_countries():
    """الحصول على قائمة الدول المتاحة"""
    if not available_countries:
        raise HTTPException(status_code=500, detail="قائمة الدول غير متاحة")
    
    # ترتيب الدول أبجدياً
    sorted_countries = sorted(available_countries)
    
    return {
        "total_countries": len(sorted_countries),
        "countries": sorted_countries[:50],  # أول 50 دولة
        "note": "يتم عرض أول 50 دولة فقط. للحصول على القائمة الكاملة استخدم /countries/all"
    }

@app.get("/countries/all")
async def get_all_countries():
    """الحصول على قائمة جميع الدول"""
    if not available_countries:
        raise HTTPException(status_code=500, detail="قائمة الدول غير متاحة")
    
    return {
        "total_countries": len(available_countries),
        "countries": sorted(available_countries)
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict_life_expectancy(input_data: PredictionInput):
    """التنبؤ بمتوقع العمر"""
    
    # التحقق من تحميل النموذج
    if model is None:
        raise HTTPException(status_code=500, detail="النموذج غير محمل بشكل صحيح")
    
    try:
        # تحضير البيانات
        prepared_data = validate_and_prepare_input(input_data)
        
        # التنبؤ
        prediction = model.predict(prepared_data)[0]
        
        # تقييم مستوى الثقة بناءً على القيم
        confidence = "عالي"
        message = f"التنبؤ لدولة {input_data.Country} في عام {input_data.Year}"
        
        # تحديد مستوى الثقة بناءً على القيم الشاذة
        if prediction < 40 or prediction > 90:
            confidence = "منخفض"
            message += " - قيم غير عادية قد تؤثر على دقة التنبؤ"
        elif 40 <= prediction < 50 or 85 < prediction <= 90:
            confidence = "متوسط"
            message += " - قيم في النطاق الحدي"
        
        # التأكد من القيمة المنطقية
        prediction = max(20, min(100, prediction))  # بين 20 و 100 سنة
        
        return PredictionOutput(
            predicted_life_expectancy=round(prediction, 2),
            confidence=confidence,
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"خطأ في عملية التنبؤ: {str(e)}"
        )

@app.get("/model/info")
async def model_info():
    """معلومات عن النموذج"""
    if model is None:
        raise HTTPException(status_code=500, detail="النموذج غير محمل")
    
    return {
        "model_type": "Linear Regression",
        "accuracy": "96.3%",
        "features_used": len(selected_features),
        "features_list": selected_features,
        "training_samples": 2350,
        "test_samples": 588,
        "average_error": "1.15 years",
        "note": "النموذج مدرب على بيانات منظمة الصحة العالمية"
    }

# تشغيل التطبيق
if __name__ == "__main__":
    import uvicorn
    print("🚀 بدء تشغيل API...")
    print("📊 النموذج جاهز للاستخدام")
    print("🌐 الوصول للوثائق: http://127.0.0.1:8000/docs")
    
    uvicorn.run(app, host="127.0.0.1", port=8000)