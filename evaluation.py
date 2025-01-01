import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model


class GestureModelEvaluator:
    def __init__(self, model_path, test_data_path):
        """تهيئة المقيّم مع مسارات النموذج والبيانات"""
        self.model = load_model(model_path)
        self.test_data = pd.read_excel(test_data_path)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # طباعة ملخص النموذج للتحقق من الأبعاد
        print("\nModel Summary:")
        self.model.summary()

    def prepare_test_data(self):
        """تجهيز بيانات الاختبار بما يتناسب مع أبعاد النموذج"""
        # فصل الميزات والتصنيفات
        X_test = self.test_data.drop('Gesture', axis=1).values
        y_test = self.label_encoder.fit_transform(self.test_data['Gesture'])

        # تطبيع البيانات
        X_test = self.scaler.fit_transform(X_test)

        # طباعة شكل البيانات للتحقق
        print(f"\nInput shape before reshaping: {X_test.shape}")

        # التأكد من أن البيانات تتناسب مع مدخلات النموذج
        expected_shape = self.model.get_layer(index=0).input_shape[1]
        if X_test.shape[1] != expected_shape:
            print(
                f"\nWarning: Input shape mismatch. Model expects {expected_shape} features, but data has {X_test.shape[1]} features.")
            # اختيار أول expected_shape من الميزات إذا كان عدد الميزات أكبر
            if X_test.shape[1] > expected_shape:
                print(f"Truncating features to match model input shape...")
                X_test = X_test[:, :expected_shape]
            else:
                raise ValueError(f"Not enough features in data. Model expects {expected_shape} features.")

        print(f"Final input shape: {X_test.shape}")
        return X_test, y_test

    def evaluate_model(self, X_test, y_test):
        """تقييم أداء النموذج"""
        # التنبؤ بالفئات
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # حساب المقاييس
        accuracy = accuracy_score(y_test, y_pred_classes)
        f1 = f1_score(y_test, y_pred_classes, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred_classes)
        class_report = classification_report(y_test, y_pred_classes)

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'y_pred': y_pred,
            'y_pred_classes': y_pred_classes
        }

    def plot_confusion_matrix(self, conf_matrix):
        """رسم مصفوفة الارتباك"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()

    def plot_class_accuracy(self, y_test, y_pred_classes):
        """رسم دقة التصنيف لكل فئة"""
        class_accuracy = {}
        for class_idx in np.unique(y_test):
            mask = y_test == class_idx
            class_accuracy[f"Class {class_idx}"] = accuracy_score(
                y_test[mask], y_pred_classes[mask]
            )

        plt.figure(figsize=(10, 6))
        plt.bar(class_accuracy.keys(), class_accuracy.values())
        plt.title('Per-Class Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Class')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('class_accuracy.png')
        plt.close()

    def save_evaluation_results(self, results, output_file="evaluation_results.txt"):
        """حفظ نتائج التقييم في ملف نصي"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=== Model Evaluation Results ===\n\n")
            f.write(f"Overall Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"Weighted F1-Score: {results['f1_score']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(results['classification_report'])


def main():
    """الدالة الرئيسية لتشغيل التقييم"""
    try:
        # تهيئة المقيم
        evaluator = GestureModelEvaluator(
            model_path="gesture_cnn_model.h5",
            test_data_path="balanced_gesture_features.xlsx"
        )

        # تجهيز البيانات وتقييم النموذج
        print("Preparing test data...")
        X_test, y_test = evaluator.prepare_test_data()

        print("Evaluating model...")
        results = evaluator.evaluate_model(X_test, y_test)

        # طباعة النتائج
        print("\n=== Evaluation Results ===")
        print(f"Overall Accuracy: {results['accuracy']:.4f}")
        print(f"Weighted F1-Score: {results['f1_score']:.4f}")
        print("\nDetailed Classification Report:")
        print(results['classification_report'])

        # رسم النتائج
        print("\nGenerating visualizations...")
        evaluator.plot_confusion_matrix(results['confusion_matrix'])
        evaluator.plot_class_accuracy(y_test, results['y_pred_classes'])

        # حفظ النتائج
        evaluator.save_evaluation_results(results)
        print("\nResults saved to evaluation_results.txt")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()