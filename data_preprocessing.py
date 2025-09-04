import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("🧹 بدء تنظيف ومعالجة البيانات...")

# تحميل البيانات
df = pd.read_csv('deathrate.csv')
print(f"✅ تم تحميل البيانات: {df.shape[0]} صف، {df.shape[1]} عمود")

# الخطوة 1: تنظيف أسماء الأعمدة (إزالة المسافات الإضافية)
print("\n1️⃣ تنظيف أسماء الأعمدة...")
df.columns = df.columns.str.strip()
print("✅ تم تنظيف أسماء الأعمدة من المسافات")

# عرض الأعمدة المنظفة
print("\n📋 أسماء الأعمدة بعد التنظيف:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. '{col}'")

# الخطوة 2: تحديد العمود المستهدف
target_column = 'Life expectancy'  # بعد إزالة المسافة الإضافية
print(f"\n🎯 العمود المستهدف: '{target_column}'")

# الخطوة 3: معالجة القيم المفقودة
print("\n2️⃣ معالجة القيم المفقودة...")

# أولاً: المتغيرات الرقمية
numeric_columns = df.select_dtypes(include=[np.number]).columns
print(f"المتغيرات الرقمية: {len(numeric_columns)} متغير")

for col in numeric_columns:
    if df[col].isnull().sum() > 0:
        missing_before = df[col].isnull().sum()
        
        # املأ القيم المفقودة بمتوسط الدولة أولاً
        df[col] = df.groupby('Country')[col].transform(
            lambda x: x.fillna(x.mean())
        )
        
        # إذا ما زالت هناك قيم مفقودة، استخدم متوسط الحالة (Status)
        if df[col].isnull().sum() > 0:
            df[col] = df.groupby('Status')[col].transform(
                lambda x: x.fillna(x.mean())
            )
        
        # إذا ما زالت هناك قيم مفقودة، استخدم المتوسط العام
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())
            
        missing_after = df[col].isnull().sum()
        print(f"  ✅ {col}: {missing_before} → {missing_after}")

# ثانياً: المتغيرات النصية
categorical_columns = df.select_dtypes(include=['object']).columns
print(f"\nالمتغيرات النصية: {len(categorical_columns)} متغير")

for col in categorical_columns:
    if df[col].isnull().sum() > 0:
        missing_before = df[col].isnull().sum()
        mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
        df[col] = df[col].fillna(mode_value)
        missing_after = df[col].isnull().sum()
        print(f"  ✅ {col}: {missing_before} → {missing_after}")

# التحقق من النتيجة النهائية
total_missing = df.isnull().sum().sum()
print(f"\n✅ إجمالي القيم المفقودة بعد المعالجة: {total_missing}")

# الخطوة 4: تحويل المتغيرات النصية إلى رقمية
print("\n3️⃣ تحويل المتغيرات النصية إلى رقمية...")

df_processed = df.copy()
encoders = {}

# Label Encoding لمتغير Status
if 'Status' in df_processed.columns:
    le_status = LabelEncoder()
    df_processed['Status'] = le_status.fit_transform(df_processed['Status'])
    encoders['Status'] = le_status
    print(f"  ✅ Status: {list(le_status.classes_)}")

# Target Encoding للدول (استخدام متوسط متوقع العمر)
if 'Country' in df_processed.columns:
    country_means = df_processed.groupby('Country')[target_column].mean()
    df_processed['Country_encoded'] = df_processed['Country'].map(country_means)
    
    # حفظ معلومات ترميز الدول
    country_encoding = country_means.to_dict()
    encoders['Country'] = country_encoding
    
    # حذف العمود الأصلي
    df_processed = df_processed.drop('Country', axis=1)
    print(f"  ✅ Country: تم ترميز {len(country_encoding)} دولة")

# الخطوة 5: فحص البيانات النهائية
print("\n4️⃣ فحص البيانات النهائية...")
print(f"الشكل النهائي: {df_processed.shape[0]} صف، {df_processed.shape[1]} عمود")
print(f"القيم المفقودة: {df_processed.isnull().sum().sum()}")

# عرض الإحصائيات الأساسية للعمود المستهدف
if target_column in df_processed.columns:
    target_stats = df_processed[target_column].describe()
    print(f"\n📊 إحصائيات {target_column}:")
    print(f"المتوسط: {target_stats['mean']:.2f}")
    print(f"الوسيط: {target_stats['50%']:.2f}")
    print(f"أقل قيمة: {target_stats['min']:.2f}")
    print(f"أعلى قيمة: {target_stats['max']:.2f}")

# الخطوة 6: حفظ البيانات المعالجة
print("\n5️⃣ حفظ البيانات...")

# حفظ البيانات المعالجة
df_processed.to_csv('data_cleaned.csv', index=False)
print("✅ تم حفظ البيانات المعالجة في: data_cleaned.csv")

# حفظ معلومات المحولات
import joblib
joblib.dump(encoders, 'encoders.joblib')
print("✅ تم حفظ معلومات المحولات في: encoders.joblib")

# حفظ اسم العمود المستهدف
with open('target_column.txt', 'w', encoding='utf-8') as f:
    f.write(target_column)
print("✅ تم حفظ اسم العمود المستهدف")

# عرض عينة من البيانات المعالجة
print(f"\n📋 عينة من البيانات المعالجة:")
print(df_processed.head(3))

print(f"\n✅ انتهاء معالجة البيانات بنجاح!")
print(f"📁 الملفات المحفوظة:")
print(f"   - data_cleaned.csv (البيانات المعالجة)")
print(f"   - encoders.joblib (معلومات التحويل)")
print(f"   - target_column.txt (اسم العمود المستهدف)")

print(f"\n🎯 الخطوة التالية: تشغيل model_development.py")
