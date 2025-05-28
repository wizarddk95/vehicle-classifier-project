import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image


def analyze_model_output(model_name, timestamp, image_paths, y_pred, y_prob, y_true, class_names, output_root="../output_analysis"):
    os.makedirs(output_root, exist_ok=True)
    model_output_dir = os.path.join(output_root, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    confidence = np.max(y_prob, axis=1)
    entropy = -np.sum(y_prob * np.log(np.clip(y_prob, 1e-15, 1)), axis=1)

    # 1. 낮은 확신도 이미지 시각화
    top_k = 16
    low_conf_idx = np.argsort(confidence)[:top_k]
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for ax, idx in zip(axes.flatten(), low_conf_idx):
        try:
            img = Image.open(image_paths[idx]).convert("RGB")
            ax.imshow(img)
        except:
            ax.set_facecolor('lightgray')
            ax.text(0.5, 0.5, "Image\nNot Found", ha='center', va='center')
        ax.set_title(f"Pred: {class_names[y_pred[idx]]}\nTrue: {class_names[y_true[idx]]}\nConf: {confidence[idx]:.2f}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(model_output_dir, f"low_confidence_{model_name}_{timestamp}.png"))
    plt.close()

    # 2. 전체 확신도 분포
    plt.figure(figsize=(8, 5))
    sns.histplot(confidence, bins=30, kde=True, color='cornflowerblue')
    plt.title(f"{model_name} 예측 확신도 분포")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(model_output_dir, f"confidence_hist_{model_name}_{timestamp}.png"))
    plt.close()

    # 3. 낮은 확신도 샘플 csv 저장
    low_conf_df = pd.DataFrame({
        'image_path': [image_paths[i] for i in low_conf_idx],
        'pred_class': [class_names[y_pred[i]] for i in low_conf_idx],
        'true_class': [class_names[y_true[i]] for i in low_conf_idx],
        'confidence': [confidence[i] for i in low_conf_idx],
        'entropy': [entropy[i] for i in low_conf_idx]
    })
    low_conf_df.to_csv(os.path.join(model_output_dir, f"low_confidence_{model_name}_{timestamp}.csv"), index=False)

    # 4. 전체 예측 결과 csv 저장
    result_df = pd.DataFrame({
        'image_path': image_paths,
        'pred_class': [class_names[i] for i in y_pred],
        'true_class': [class_names[i] for i in y_true],
        'confidence': confidence,
        'entropy': entropy,
        'correct': (np.array(y_pred) == np.array(y_true))
    })
    result_df.to_csv(os.path.join(model_output_dir, f"eval_result_{model_name}_{timestamp}.csv"), index=False)

    # 5. 클래스별 평균 확신도 barplot (상/하위 20개 비교)
    df = pd.DataFrame({'class': [class_names[i] for i in y_pred], 'confidence': confidence})
    mean_conf = df.groupby('class').mean().sort_values(by='confidence', ascending=True)

    # 하위 20개와 상위 20개 클래스 추출
    bottom_20 = mean_conf.head(20)
    top_20 = mean_conf.tail(20)

    # 시각화
    fig, axs = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    sns.barplot(x=bottom_20.index, y=bottom_20['confidence'], ax=axs[0], palette='magma')
    axs[0].set_title(f"{model_name} 하위 20 클래스 평균 확신도")
    axs[0].tick_params(axis='x', rotation=90)

    sns.barplot(x=top_20.index, y=top_20['confidence'], ax=axs[1], palette='viridis')
    axs[1].set_title(f"{model_name} 상위 20 클래스 평균 확신도")
    axs[1].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.savefig(os.path.join(model_output_dir, f"classwise_confidence_top_bottom_{model_name}_{timestamp}.png"))
    plt.close()


