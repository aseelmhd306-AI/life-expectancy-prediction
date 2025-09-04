import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
import joblib
import warnings
warnings.filterwarnings('ignore')

# تعيين الخط للرسوم البيانية
plt.rcParams['figure.figsize'] = (10, 6)
sns.set_style("whitegrid")

print("🤖 بدء بناء نموذج التنبؤ بمتوقع العمر...")

# الخطوة 1: تحميل البيانات المعالجة
df = pd.read_csv('data_cleaned.csv')
with open('target_column.txt', 'r', encoding='utf-8') as f:
    target_column = f.read().strip()

print(f"✅ تم تحميل البيانات: {df.shape}")
print(f"🎯 العمود المستهدف: '{target_column}'")

# الخطوة 2: تحليل الارتباط
print(f"\n1️⃣ تحليل الارتباط مع متوقع العمر...")

# حساب الارتباط مع العمود المستهدف
correlations = df.corr()[target_column].sort_values(key=abs, ascending=False)
print(f"\n📊 أقوى الارتباطات مع {target_column}:")

for var, corr in correlations.items():
    if var != target_column:
        direction = "موجب" if corr > 0 else "سالب"
        strength = "قوي" if abs(corr) > 0.7 else "متوسط" if abs(corr) > 0.4 else "ضعيف"
        print(f"  {var}: {corr:.3f} ({direction} - {strength})")

# رسم خريطة الارتباط
plt.figure(figsize=(15, 12))
correlation_matrix = df.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # إخفاء النصف العلوي
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0, 
            fmt='.2f', square=True, cbar_kws={"shrink": .8})
plt.title('خريطة الارتباط بين المتغيرات')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# الخطوة 3: تحضير البيانات للنمذجة
print(f"\n2️⃣ تحضير البيانات للنمذجة...")

X = df.drop(target_column, axis=1)
y = df[target_column]

print(f"عدد المتغيرات المستقلة: {X.shape[1]}")
print(f"عدد العينات: {X.shape[0]}")

# تقسيم البيانات (80% للتدريب، 20% للاختبار)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"بيانات التدريب: {X_train.shape[0]} عينة")
print(f"بيانات الاختبار: {X_test.shape[0]} عينة")

# الخطوة 4: النموذج الأساسي
print(f"\n3️⃣ بناء النموذج الأساسي...")

baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)

y_pred_baseline = baseline_model.predict(X_test)

mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
mse_baseline = mean_squared_error(y_test, y_pred_baseline)
rmse_baseline = np.sqrt(mse_baseline)
r2_baseline = r2_score(y_test, y_pred_baseline)

print(f"📈 أداء النموذج الأساسي (مع جميع المتغيرات):")
print(f"  MAE: {mae_baseline:.3f} سنة")
print(f"  RMSE: {rmse_baseline:.3f} سنة") 
print(f"  R²: {r2_baseline:.3f} ({r2_baseline*100:.1f}%)")

# الخطوة 5: Backward Elimination
print(f"\n4️⃣ تطبيق Backward Elimination...")

def backward_elimination(X, y, significance_level=0.05):
    """تطبيق Backward Elimination لاختيار أهم المتغيرات"""
    X_with_const = sm.add_constant(X)
    features = X_with_const.columns.tolist()
    removed_features = []
    
    print("بدء عملية الحذف التدريجي...")
    
    while True:
        model = sm.OLS(y, X_with_const[features]).fit()
        p_values = model.pvalues
        max_p_value = p_values.max()
        
        if max_p_value > significance_level:
            feature_with_max_p = p_values.idxmax()
            if feature_with_max_p != 'const':
                features.remove(feature_with_max_p)
                removed_features.append(feature_with_max_p)
                print(f"  ❌ حذف: {feature_with_max_p} (p-value: {max_p_value:.4f})")
        else:
            break
    
    if 'const' in features:
        features.remove('const')
    
    print(f"\n✅ انتهاء الحذف التدريجي")
    print(f"المتغيرات المحذوفة ({len(removed_features)}): {removed_features}")
    print(f"المتغيرات المتبقية ({len(features)}): {features}")
    
    return features, model

selected_features, final_ols_model = backward_elimination(X_train, y_train)

# الخطوة 6: النموذج المحسن
print(f"\n5️⃣ بناء النموذج المحسن...")

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

improved_model = LinearRegression()
improved_model.fit(X_train_selected, y_train)

y_pred_improved = improved_model.predict(X_test_selected)

mae_improved = mean_absolute_error(y_test, y_pred_improved)
mse_improved = mean_squared_error(y_test, y_pred_improved)
rmse_improved = np.sqrt(mse_improved)
r2_improved = r2_score(y_test, y_pred_improved)

print(f"📈 أداء النموذج المحسن:")
print(f"  MAE: {mae_improved:.3f} سنة")
print(f"  RMSE: {rmse_improved:.3f} سنة")
print(f"  R²: {r2_improved:.3f} ({r2_improved*100:.1f}%)")

# الخطوة 7: مقارنة النماذج
print(f"\n6️⃣ مقارنة النماذج...")

comparison = pd.DataFrame({
    'المقياس': ['MAE (سنة)', 'RMSE (سنة)', 'R² (%)', 'عدد المتغيرات'],
    'النموذج الأساسي': [mae_baseline, rmse_baseline, r2_baseline*100, len(X.columns)],
    'النموذج المحسن': [mae_improved, rmse_improved, r2_improved*100, len(selected_features)]
})

print(comparison.round(3))

# اختيار النموذج الأفضل
if r2_improved >= r2_baseline - 0.01:  # إذا كان الأداء متقارب، نختار الأبسط
    final_model = improved_model
    final_features = selected_features
    final_r2 = r2_improved
    final_mae = mae_improved
    model_type = "محسن"
else:
    final_model = baseline_model
    final_features = list(X.columns)
    final_r2 = r2_baseline
    final_mae = mae_baseline
    model_type = "أساسي"

print(f"\n🏆 النموذج المختار: {model_type}")
print(f"دقة النموذج: {final_r2:.3f} ({final_r2*100:.1f}%)")
print(f"متوسط الخطأ: {final_mae:.3f} سنة")

# الخطوة 8: حفظ النموذج
print(f"\n7️⃣ حفظ النموذج النهائي...")

joblib.dump(final_model, 'life_expectancy_model.joblib')
joblib.dump(final_features, 'selected_features.joblib')

print("✅ تم حفظ الملفات:")
print("  - life_expectancy_model.joblib (النموذج)")
print("  - selected_features.joblib (قائمة المتغيرات)")

# الخطوة 9: اختبار النموذج
print(f"\n8️⃣ اختبار النموذج...")

# اختيار عينة عشوائية من بيانات الاختبار
sample_idx = np.random.choice(len(X_test), 3)
if model_type == "محسن":
    sample_pred = final_model.predict(X_test_selected.iloc[sample_idx])
else:
    sample_pred = final_model.predict(X_test.iloc[sample_idx])

print("🧪 اختبارات عشوائية:")
for i, idx in enumerate(sample_idx):
    actual = y_test.iloc[idx]
    predicted = sample_pred[i]
    error = abs(actual - predicted)
    print(f"  العينة {i+1}: فعلي={actual:.1f}, متوقع={predicted:.1f}, الخطأ={error:.1f}")

# الخطوة 10: الرسوم البيانية
print(f"\n9️⃣ إنشاء الرسوم البيانية...")

# رسم مقارنة النماذج
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. مقارنة المقاييس
metrics = ['MAE', 'RMSE', 'R²']
baseline_scores = [mae_baseline, rmse_baseline, r2_baseline]
improved_scores = [mae_improved, rmse_improved, r2_improved]

x_pos = np.arange(len(metrics))
width = 0.35

ax1.bar(x_pos - width/2, baseline_scores, width, label='أساسي', alpha=0.7, color='lightblue')
ax1.bar(x_pos + width/2, improved_scores, width, label='محسن', alpha=0.7, color='lightgreen')
ax1.set_xlabel('المقاييس')
ax1.set_ylabel('القيم')
ax1.set_title('مقارنة أداء النماذج')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(metrics)
ax1.legend()

# 2. القيم الفعلية مقابل المتوقعة
if model_type == "محسن":
    y_pred_final = final_model.predict(X_test_selected)
else:
    y_pred_final = final_model.predict(X_test)

ax2.scatter(y_test, y_pred_final, alpha=0.6, color='blue')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel('القيم الفعلية')
ax2.set_ylabel('القيم المتوقعة')
ax2.set_title(f'الفعلي مقابل المتوقع - النموذج {model_type}')

# 3. توزيع الأخطاء
residuals = y_test - y_pred_final
ax3.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
ax3.set_xlabel('الأخطاء (الفعلي - المتوقع)')
ax3.set_ylabel('التكرار')
ax3.set_title('توزيع أخطاء النموذج')
ax3.axvline(residuals.mean(), color='red', linestyle='--', 
           label=f'المتوسط: {residuals.mean():.2f}')
ax3.legend()

# 4. أهمية المتغيرات
feature_importance = pd.DataFrame({
    'المتغير': final_features,
    'المعامل': final_model.coef_
})
feature_importance['الأهمية'] = abs(feature_importance['المعامل'])
feature_importance = feature_importance.sort_values('الأهمية', ascending=True).tail(10)

ax4.barh(range(len(feature_importance)), feature_importance['الأهمية'], 
         color='lightcoral', alpha=0.7)
ax4.set_yticks(range(len(feature_importance)))
ax4.set_yticklabels(feature_importance['المتغير'])
ax4.set_xlabel('الأهمية (القيمة المطلقة للمعامل)')
ax4.set_title('أهم 10 متغيرات في النموذج')

plt.tight_layout()
plt.savefig('model_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# الخطوة 11: تقرير نهائي
print(f"\n🔟 إنشاء التقرير النهائي...")

report = f"""
تقرير نموذج التنبؤ بمتوقع العمر
{"="*50}

📊 معلومات البيانات:
- إجمالي العينات: {len(df):,}
- بيانات التدريب: {len(X_train):,} عينة
- بيانات الاختبار: {len(X_test):,} عينة
- المتغيرات الأصلية: {len(X.columns)}
- المتغيرات المختارة: {len(final_features)}

🎯 النموذج المختار: {model_type}
- دقة النموذج (R²): {final_r2:.3f} ({final_r2*100:.1f}%)
- متوسط الخطأ المطلق: {final_mae:.3f} سنة
- الجذر التربيعي لمتوسط مربع الخطأ: {rmse_improved if model_type == "محسن" else rmse_baseline:.3f} سنة

📈 تفسير النتائج:
- النموذج يفسر {final_r2*100:.1f}% من التباين في متوقع العمر
- متوسط خطأ التنبؤ: ±{final_mae:.1f} سنة
- النموذج {"ممتاز" if final_r2 > 0.9 else "جيد جداً" if final_r2 > 0.8 else "جيد" if final_r2 > 0.7 else "مقبول"}

🔍 أهم المتغيرات المؤثرة:
"""

# إضافة أهم المتغيرات
importance_df = pd.DataFrame({
    'المتغير': final_features,
    'المعامل': final_model.coef_
}).sort_values(key=lambda x: abs(x), by='المعامل', ascending=False).head(5)

for _, row in importance_df.iterrows():
    effect = "يزيد" if row['المعامل'] > 0 else "يقلل"
    report += f"- {row['المتغير']}: {effect} متوقع العمر بـ {abs(row['المعامل']):.3f} سنة\n"

report += f"""
💾 الملفات المحفوظة:
- life_expectancy_model.joblib: النموذج النهائي
- selected_features.joblib: قائمة المتغيرات المختارة
- encoders.joblib: معلومات تحويل البيانات
- data_cleaned.csv: البيانات المعالجة
- correlation_heatmap.png: خريطة الارتباط
- model_analysis.png: تحليل النموذج

🚀 جاهز لبناء API!
"""

with open('model_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("✅ تم حفظ التقرير في: model_report.txt")

print(f"\n🎉 تم الانتهاء من بناء النموذج بنجاح!")
print(f"🎯 الخطوة التالية: بناء API للنموذج")
print(f"📊 دقة النموذج: {final_r2*100:.1f}%")
print(f"📁 جميع الملفات محفوظة ومجهزة لبناء API")