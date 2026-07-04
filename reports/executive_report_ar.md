# تقرير تنفيذي: التنبؤ بنتيجة فحص طيف التوحد للبالغين

## هدف المشروع

يهدف هذا المشروع إلى بناء نموذج Machine Learning يتنبأ باحتمالية ظهور نتيجة فحص إيجابية لطيف التوحد لدى البالغين، اعتمادًا على بيانات استبيان AQ-10 وبعض المؤشرات الديموغرافية والطبية المتاحة في مجموعة بيانات UCI.

> تنبيه مهني: هذا المشروع تعليمي وتحليلي لأغراض Portfolio، ولا يمثل أداة تشخيص طبي. التشخيص لا يتم إلا بواسطة مختصين سريريين.

## البيانات

- المصدر: UCI Machine Learning Repository - Autism Screening Adult.
- عدد السجلات بعد التنظيف: 680.
- عدد نتائج ASD: 182.
- عدد نتائج No ASD: 498.
- نسبة النتائج الإيجابية: 26.8%.
- توزيع الهدف: {'No ASD': 498, 'ASD': 182}.

## ما تم تنفيذه

1. تحميل ملف ARFF الخام من UCI.
2. تحويل البيانات إلى DataFrame وتنظيف أسماء الأعمدة والقيم المفقودة.
3. استبعاد العمر غير المنطقي والقيم المكررة.
4. استبعاد عمود `result` من التدريب لأنه مجموع مباشر للأسئلة وقد يسبب تسريبًا للمعلومة.
5. بناء رسوم EDA توضح توزيع الهدف، العمر، إجابات AQ-10، الارتباطات، والفروق حسب الدولة.
6. تدريب ثلاثة نماذج: Logistic Regression وRandom Forest وGradient Boosting.
7. تقييم النماذج باستخدام Accuracy وPrecision وRecall وF1 وROC-AUC مع Cross Validation.
8. حفظ أفضل نموذج ونتائج التنبؤات وأهمية الخصائص.

## أفضل نموذج

- النموذج المختار: Logistic Regression.
- Test Accuracy: 1.000.
- Test Precision: 1.000.
- Test Recall: 1.000.
- Test F1: 1.000.
- Test ROC-AUC: 1.000.

ملاحظة تقييمية: الأرقام مرتفعة جدًا لأن الهدف مرتبط بقوة بإجابات AQ-10 نفسها. تم استبعاد عمود `result` لتقليل التسريب المباشر، لكن النتائج يجب تفسيرها كدليل على Workflow قابل للتكرار وليس كدليل على أداء تشخيصي سريري.

## أهم الخصائص

| feature | importance_mean | importance_std |
|---|---:|---:|
| a4_score | 0.0117 | 0.0042 |
| a5_score | 0.0112 | 0.0035 |
| a9_score | 0.0098 | 0.0025 |
| a1_score | 0.0093 | 0.0029 |
| a2_score | 0.0078 | 0.0019 |
| a6_score | 0.0077 | 0.0035 |
| a3_score | 0.0075 | 0.0024 |
| a7_score | 0.0063 | 0.0024 |

## المخرجات

- Dataset نظيف: `data/processed/autism_adult_cleaned.csv`
- مقاييس النماذج: `reports/tables/model_performance.csv`
- مصفوفة الالتباس: `reports/tables/confusion_matrix.csv`
- أهمية الخصائص: `reports/tables/feature_importance.csv`
- تنبؤات عينة: `reports/tables/sample_predictions.csv`
- النموذج المحفوظ: `models/best_autism_screening_model.joblib`
