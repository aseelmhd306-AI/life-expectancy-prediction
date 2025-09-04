import pandas as pd
import numpy as np

print("🚀 بدء فحص البيانات...")

# محاولة تحميل الملف بأسماء مختلفة محتملة
file_found = False
possible_names = [
    'deathrate.csv',
    'deathrate.xlsx', 
    'Life Expectancy Data.csv',
    'Life_Expectancy_Data.csv',
    'life_expectancy.csv',
    'WHO.csv'
]

df = None
file_name = ""

for name in possible_names:
    try:
        if name.endswith('.csv'):
            df = pd.read_csv(name)
        else:
            df = pd.read_excel(name)
        file_name = name
        file_found = True
        print(f"✅ تم العثور على الملف: {name}")
        break
    except:
        continue

if not file_found:
    print("❌ لم يتم العثور على ملف البيانات!")
    print("الملفات الموجودة في المجلد:")
    import os
    files = os.listdir('.')
    for f in files:
        if f.endswith(('.csv', '.xlsx', '.xls')):
            print(f"  - {f}")
    
    print("\nإذا كان اسم ملفك مختلف، غير الاسم في الكود أعلاه")
    exit()

# عرض معلومات أساسية
print(f"\n📊 معلومات الملف: {file_name}")
print(f"عدد الصفوف: {df.shape[0]:,}")
print(f"عدد الأعمدة: {df.shape[1]}")

print(f"\n📋 أسماء الأعمدة ({len(df.columns)} عمود):")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. '{col}'")

# البحث عن العمود المستهدف
print(f"\n🔍 البحث عن عمود متوقع العمر...")
target_candidates = []
for col in df.columns:
    col_clean = col.lower().strip()
    if any(word in col_clean for word in ['life', 'expectancy', 'lifeexpectancy']):
        target_candidates.append(col)
        print(f"✅ عثرت على مرشح: '{col}'")

if not target_candidates:
    print("❌ لم أعثر على عمود متوقع العمر تلقائياً")
    print("يرجى فحص قائمة الأعمدة أعلاه وإخباري بالرقم الصحيح")
else:
    target_col = target_candidates[0]
    print(f"🎯 سأستخدم العمود: '{target_col}'")

# عرض عينة من البيانات
print(f"\n📋 أول 3 صفوف من البيانات:")
print(df.head(3))

print(f"\n🕳️ فحص القيم المفقودة:")
missing = df.isnull().sum()
total_missing = missing.sum()

if total_missing == 0:
    print("✅ ممتاز! لا توجد قيم مفقودة")
else:
    print(f"إجمالي القيم المفقودة: {total_missing:,}")
    for col in df.columns:
        if missing[col] > 0:
            percent = (missing[col] / len(df)) * 100
            print(f"  - {col}: {missing[col]:,} ({percent:.1f}%)")

# فحص المكرر
duplicates = df.duplicated().sum() 
print(f"\n📋 الصفوف المكررة: {duplicates:,}")

# فحص أنواع البيانات
print(f"\n📊 أنواع البيانات:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
text_cols = df.select_dtypes(include=['object']).columns

print(f"أعمدة رقمية ({len(numeric_cols)}): {', '.join(numeric_cols)}")
print(f"أعمدة نصية ({len(text_cols)}): {', '.join(text_cols)}")

print(f"\n✅ انتهاء الفحص الأولي!")
print(f"ملخص: {len(df)} صف، {len(df.columns)} عمود، {total_missing} قيمة مفقودة، {duplicates} صف مكرر")

# حفظ اسم العمود المستهدف للاستخدام لاحقاً
if target_candidates:
    with open('target_column.txt', 'w', encoding='utf-8') as f:
        f.write(target_candidates[0])
    print(f"💾 تم حفظ اسم العمود المستهدف: {target_candidates[0]}")