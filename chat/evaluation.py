# model_evaluator.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from model_training import GestureModelBuilder


class GestureModelEvaluator:
    """
    تقييم شامل لنموذج التعرف على الإيماءات مع التركيز على:
    - تقييم متوازن للفئات
    - مقاييس متعددة للأداء
    - تحليل بصري للنتائج
    """

    def __init__(self, model):
        self.model = model

    def evaluate_model(self, X_test, y_test, class_names):
        """تقييم شامل للنموذج"""
        # الحصول على التنبؤات
        y_pred_prob = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)

        # حساب المقاييس الأساسية
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_prob)

        # طباعة النتائج
        self._print_evaluation_results(metrics, y_test, y_pred, class_names)

        # رسم المخططات
        self._plot_evaluation_charts(y_test, y_pred, y_pred_prob, class_names)

        return metrics

    def _calculate_metrics(self, y_true, y_pred, y_pred_prob):
        """حساب مقاييس الأداء المختلفة"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'per_class_f1': f1_score(y_true, y_pred, average=None)
        }

    def _print_evaluation_results(self, metrics, y_true, y_pred, class_names):
        """طباعة نتائج التقييم"""
        print("\n=== تقييم شامل للنموذج ===")
        print(f"\nالدقة الكلية: {metrics['accuracy']:.4f}")
        print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
        print(f"F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
        print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (Macro): {metrics['recall_macro']:.4f}")

        print("\nأداء كل فئة:")
        for i, class_name in enumerate(class_names):
            print(f"\nالفئة {class_name}:")
            print(f"F1 Score: {metrics['per_class_f1'][i]:.4f}")

        print("\nتقرير التصنيف المفصل:")
        print(classification_report(y_true, y_pred, target_names=class_names))

    def _plot_evaluation_charts(self, y_true, y_pred, y_pred_prob, class_names):
        """رسم مخططات التقييم"""
        # مصفوفة الارتباك
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('مصفوفة الارتباك')
        plt.ylabel('القيمة الحقيقية')
        plt.xlabel('التنبؤ')
        plt.show()

        # منحنيات ROC
        plt.figure(figsize=(10, 8))
        y_true_bin = tf.keras.utils.to_categorical(y_true)
        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
            plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc(fpr, tpr):.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('معدل الإيجابيات الخاطئة')
        plt.ylabel('معدل الإيجابيات الصحيحة')
        plt.title('منحنيات ROC')
        plt.legend()
        plt.show()

    def plot_training_history(self, history):
        """تحليل تاريخ التدريب"""
        plt.figure(figsize=(12, 4))

        # منحنى الدقة
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='تدريب')
        plt.plot(history.history['val_accuracy'], label='تحقق')
        plt.title('دقة النموذج')
        plt.xlabel('Epoch')
        plt.ylabel('الدقة')
        plt.legend()

        # منحنى الخسارة
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='تدريب')
        plt.plot(history.history['val_loss'], label='تحقق')
        plt.title('خسارة النموذج')
        plt.xlabel('Epoch')
        plt.ylabel('الخسارة')
        plt.legend()

        plt.tight_layout()
        plt.show()

        # تحليل النتائج
        print("\n=== تحليل تاريخ التدريب ===")
        best_acc = max(history.history['val_accuracy'])
        best_epoch = history.history['val_accuracy'].index(best_acc)
        print(f"أفضل دقة في التحقق: {best_acc:.4f} (Epoch {best_epoch + 1})")

        final_loss = history.history['val_loss'][-1]
        print(f"الخسارة النهائية في التحقق: {final_loss:.4f}")


# مثال على الاستخدام
if __name__ == "__main__":
    model = tf.keras.models.load_model('gesture_model.h5')
    evaluator = GestureModelEvaluator(model)

    builder = GestureModelBuilder()
    _, X_test, _, y_test = builder.prepare_data("balanced_gesture_features.xlsx")

    # تعديل عدد الفئات ليتناسب مع البيانات (فئتين فقط)
    class_names = [str(i) for i in range(4)]
    metrics = evaluator.evaluate_model(X_test, y_test, class_names)
