# مشروع التنبؤ بمتوقع العمر 🔮

## نظرة عامة
هذا المشروع يستخدم الذكاء الاصطناعي للتنبؤ بمتوقع العمر بناءً على المؤشرات الصحية والاقتصادية. النموذج مدرب على بيانات منظمة الصحة العالمية ويحقق دقة 96.3%.

## 📊 إنجازات المشروع
- **دقة النموذج: 96.3%** - ممتاز
- **متوسط الخطأ: 1.15 سنة** - دقة عالية جداً  
- **تقليل المتغيرات: من 21 إلى 9** - نموذج مُحسَّن
- **API جاهز للاستخدام** - واجهة تفاعلية

## 🛠️ التقنيات المستخدمة
- **Python 3.x**
- **scikit-learn** - بناء النموذج
- **pandas & numpy** - معالجة البيانات
- **FastAPI** - بناء API
- **matplotlib & seaborn** - الرسوم البيانية
- **statsmodels** - التحليل الإحصائي

## 📁 هيكل المشروع
```
life_expectancy_project/
├── data_exploration.py          # استكشاف البيانات
├── data_preprocessing.py        # معالجة البيانات
├── model_development.py         # بناء النموذج
├── main.py                      # API
├── deathrate.csv               # البيانات الأصلية
├── data_cleaned.csv            # البيانات المعالجة
├── life_expectancy_model.joblib # النموذج المحفوظ
├── selected_features.joblib     # المتغيرات المختارة
├── encoders.joblib             # معلومات التحويل
├── target_column.txt           # اسم العمود المستهدف
├── model_report.txt            # تقرير النموذج
├── correlation_heatmap.png     # خريطة الارتباط
├── model_analysis.png          # تحليل النموذج
└── README.md                   # هذا الملف
```

## 🚀 طريقة التشغيل

### 1. إعداد البيئة
```bash
# إنشاء البيئة الافتراضية
python -m venv venv

# تفعيل البيئة
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# تثبيت المكتبات
pip install -r requirements.txt
```

### 2. تشغيل التحليل (اختياري)
```bash
# استكشاف البيانات
python data_exploration.py

# معالجة البيانات  
python data_preprocessing.py

# بناء النموذج
python model_development.py
```

### 3. تشغيل API
```bash
python main.py
```
أو
```bash
uvicorn main:app --reload
```

### 4. الوصول للAPI
- **الوثائق التفاعلية:** http://127.0.0.1:8000/docs
- **الصفحة الرئيسية:** http://127.0.0.1:8000/
- **فحص الحالة:** http://127.0.0.1:8000/health

## 📖 استخدام API

### الوظائف المتاحة:

#### 1. التنبؤ بمتوقع العمر
```http
POST /predict
```

**مثال على البيانات المطلوبة:**
```json
{
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
```

**الاستجابة:**
```json
{
    "predicted_life_expectancy": 74.5,
    "confidence": "عالي",
    "message": "التنبؤ لدولة Jordan في عام 2015"
}
```

#### 2. قائمة الدول المتاحة
```http
GET /countries
```

#### 3. معلومات النموذج
```http
GET /model/info  
```

#### 4. فحص حالة API
```http
GET /health
```

## 🧪 اختبار API

### استخدام curl:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
    "Country": "Jordan",
    "Year": 2015,
    "Adult_Mortality": 150,
    "infant_deaths": 5,
    "percentage_expenditure": 500.0,
    "under_five_deaths": 6,
    "Diphtheria": 95.0,
    "HIV_AIDS": 0.1,
    "Schooling": 13.5
}'
```

### استخدام Python:
```python
import requests

url = "http://127.0.0.1:8000/predict"
data = {
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

response = requests.post(url, json=data)
print(response.json())
```

## 📊 أداء النموذج

### المقاييس:
- **R² Score:** 96.3% (ممتاز)
- **Mean Absolute Error:** 1.15 سنة
- **Root Mean Square Error:** 1.79 سنة

### المتغيرات المستخدمة (9 متغيرات):
1. **Country_encoded** - ترميز الدولة
2. **Schooling** - سنوات التعليم
3. **Adult Mortality** - وفيات البالغين
4. **HIV/AIDS** - معدل HIV/AIDS
5. **Diphtheria** - تطعيم الدفتيريا
6. **Year** - السنة
7. **infant deaths** - وفيات الرضع
8. **under-five deaths** - وفيات تحت 5 سنوات
9. **percentage expenditure** - الإنفاق على الصحة

## 🎯 الخلاصة والتوصيات

### نقاط القوة:
✅ دقة عالية جداً (96.3%)  
✅ خطأ منخفض (1.15 سنة)  
✅ نموذج مُبسَّط (9 متغيرات فقط)  
✅ API سهل الاستخدام  

### التحسينات المستقبلية:
- إضافة المزيد من البيانات الحديثة
- تجربة نماذج أخرى (Random Forest, XGBoost)
- إضافة ميزة التنبؤ بالفترات الزمنية
- تحسين واجهة المستخدم
