"""
階段2: 分類器訓練工具
從收集的數據訓練分類模型，並評估性能
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib


def load_training_data(session_dirs: list, label_map: dict):
    """
    從多個 session 資料夾載入訓練數據

    Args:
        session_dirs: session 資料夾路徑列表
        label_map: session 名稱到標籤的映射，例如 {'positive_samples': 1, 'negative_samples': 0}

    Returns:
        X, y: 特徵矩陣和標籤向量
    """
    all_features = []
    all_labels = []

    for session_dir in session_dirs:
        features_csv = os.path.join(session_dir, 'features.csv')
        if not os.path.exists(features_csv):
            print(f"Warning: {features_csv} 不存在，跳過")
            continue

        # 從路徑推斷標籤
        session_name = os.path.basename(session_dir)
        label = label_map.get(session_name, None)

        if label is None:
            # 嘗試從父目錄名稱推斷
            parent_name = os.path.basename(os.path.dirname(session_dir))
            label = label_map.get(parent_name, None)

        if label is None:
            print(f"Warning: 無法推斷 {session_dir} 的標籤，跳過")
            continue

        df = pd.read_csv(features_csv)

        # 選擇特徵欄位
        feature_cols = ['peak', 'rms', 'crest', 'zcr', 'duration_ms',
                       'centroid_hz', 'rolloff85_hz', 'bandwidth_hz', 'dominant_hz']

        X_session = df[feature_cols].values
        y_session = np.full(len(X_session), label)

        all_features.append(X_session)
        all_labels.append(y_session)

        print(f"載入 {session_dir}: {len(X_session)} 樣本, label={label}")

    if not all_features:
        raise ValueError("沒有載入任何數據！")

    X = np.vstack(all_features)
    y = np.hstack(all_labels)

    return X, y


def train_random_forest(X_train, y_train, X_test, y_test):
    """訓練隨機森林分類器"""
    print("\n=== 訓練隨機森林 ===")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)

    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    print(f"訓練集準確率: {train_score:.4f}")
    print(f"測試集準確率: {test_score:.4f}")

    y_pred = clf.predict(X_test)
    print("\n分類報告:")
    print(classification_report(y_test, y_pred))

    # 特徵重要性
    feature_names = ['peak', 'rms', 'crest', 'zcr', 'duration_ms',
                    'centroid_hz', 'rolloff85_hz', 'bandwidth_hz', 'dominant_hz']
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\n特徵重要性:")
    for i in range(len(feature_names)):
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

    return clf, y_pred


def train_svm(X_train, y_train, X_test, y_test):
    """訓練 SVM 分類器"""
    print("\n=== 訓練 SVM ===")
    clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    clf.fit(X_train, y_train)

    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    print(f"訓練集準確率: {train_score:.4f}")
    print(f"測試集準確率: {test_score:.4f}")

    y_pred = clf.predict(X_test)
    print("\n分類報告:")
    print(classification_report(y_test, y_pred))

    return clf, y_pred


def plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix"):
    """繪製混淆矩陣"""
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()

    classes = ['Negative (0)', 'Positive (1)']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()


def plot_roc_curve(clf, X_test, y_test, title="ROC Curve"):
    """繪製 ROC 曲線"""
    if hasattr(clf, 'predict_proba'):
        y_score = clf.predict_proba(X_test)[:, 1]
    else:
        y_score = clf.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()


def find_optimal_threshold(clf, X_train, y_train):
    """
    找出最佳閾值（針對 RMS 單特徵分類器）
    """
    if X_train.shape[1] == 1:
        # 單特徵情況：RMS
        rms_values = X_train[:, 0]
        labels = y_train

        # 嘗試不同閾值
        thresholds = np.linspace(rms_values.min(), rms_values.max(), 100)
        best_acc = 0
        best_threshold = 0

        for thr in thresholds:
            y_pred = (rms_values > thr).astype(int)
            acc = (y_pred == labels).mean()
            if acc > best_acc:
                best_acc = acc
                best_threshold = thr

        print(f"\n最佳 RMS 閾值: {best_threshold:.2f} (準確率: {best_acc:.4f})")
        return best_threshold

    return None


def export_model_for_esp32(clf, model_type, output_path):
    """
    匯出模型參數為 ESP32 可用的 C 代碼（階段3）
    目前僅支援簡單閾值模型
    """
    print(f"\n匯出模型到 {output_path}")

    if model_type == 'threshold':
        # 假設 clf 是一個簡單的閾值模型
        # 這裡需要根據實際情況調整
        pass
    else:
        print("警告: 複雜模型匯出需要 TensorFlow Lite Micro 或手動轉換")
        print("建議使用 Edge Impulse 平台進行模型訓練與部署")


def main():
    parser = argparse.ArgumentParser(description='訓練撞擊分類器')
    parser.add_argument('--positive', nargs='+', required=True, help='正樣本資料夾路徑')
    parser.add_argument('--negative', nargs='+', required=True, help='負樣本資料夾路徑')
    parser.add_argument('--model', choices=['rf', 'svm', 'both'], default='both', help='模型類型')
    parser.add_argument('--output', default='models', help='模型輸出目錄')
    parser.add_argument('--test-size', type=float, default=0.2, help='測試集比例')

    args = parser.parse_args()

    # 載入數據
    label_map = {}
    for path in args.positive:
        label_map[os.path.basename(path)] = 1
    for path in args.negative:
        label_map[os.path.basename(path)] = 0

    all_dirs = args.positive + args.negative
    X, y = load_training_data(all_dirs, label_map)

    print(f"\n總樣本數: {len(y)}")
    print(f"正樣本: {(y == 1).sum()}, 負樣本: {(y == 0).sum()}")

    # 分割訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    print(f"訓練集: {len(y_train)}, 測試集: {len(y_test)}")

    os.makedirs(args.output, exist_ok=True)

    # 訓練模型
    if args.model in ['rf', 'both']:
        clf_rf, y_pred_rf = train_random_forest(X_train, y_train, X_test, y_test)

        # 儲存模型
        model_path = os.path.join(args.output, 'random_forest.joblib')
        joblib.dump(clf_rf, model_path)
        print(f"\n隨機森林模型已儲存至: {model_path}")

        # 繪製混淆矩陣和 ROC 曲線
        plot_confusion_matrix(y_test, y_pred_rf, "Random Forest - Confusion Matrix")
        plt.savefig(os.path.join(args.output, 'rf_confusion_matrix.png'))

        plot_roc_curve(clf_rf, X_test, y_test, "Random Forest - ROC Curve")
        plt.savefig(os.path.join(args.output, 'rf_roc_curve.png'))

    if args.model in ['svm', 'both']:
        clf_svm, y_pred_svm = train_svm(X_train, y_train, X_test, y_test)

        # 儲存模型
        model_path = os.path.join(args.output, 'svm.joblib')
        joblib.dump(clf_svm, model_path)
        print(f"\nSVM 模型已儲存至: {model_path}")

        # 繪製混淆矩陣和 ROC 曲線
        plot_confusion_matrix(y_test, y_pred_svm, "SVM - Confusion Matrix")
        plt.savefig(os.path.join(args.output, 'svm_confusion_matrix.png'))

        plot_roc_curve(clf_svm, X_test, y_test, "SVM - ROC Curve")
        plt.savefig(os.path.join(args.output, 'svm_roc_curve.png'))

    # 尋找最佳閾值（如果只使用 RMS 特徵）
    find_optimal_threshold(clf_rf if 'clf_rf' in locals() else clf_svm, X_train, y_train)

    print("\n訓練完成！")
    print(f"模型和圖表已儲存至: {args.output}/")

    plt.show()


if __name__ == '__main__':
    main()
