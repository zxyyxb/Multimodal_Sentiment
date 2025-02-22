import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def load_and_preprocess_data(
        txt_dir="人工智能实验5/src/data/",  # 文本目录
        img_dir="人工智能实验5/src/data/",  # 图片目录
        train_file="人工智能实验5/src/train.txt",
        vocab_size=5000,
        max_len=50,
        img_height=224,
        img_width=224
):
    df = pd.read_csv(train_file, sep=",", header=0, names=["guid", "tag"])

    label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
    df["label"] = df["tag"].map(label_mapping)

    texts = []
    images = []
    labels = []

    for idx, row in df.iterrows():
        guid = str(row["guid"])
        label = row["label"]

        txt_path = os.path.join(txt_dir, guid + ".txt")
        img_path = os.path.join(img_dir, guid + ".jpg")

        if not os.path.isfile(txt_path):
            print(f"[Warning] 文本文件不存在: {txt_path}, 跳过该样本")
            continue
        if not os.path.isfile(img_path):
            print(f"[Warning] 图片文件不存在: {img_path}, 跳过该样本")
            continue

        with open(txt_path, "r", encoding="ascii", errors="replace") as f:
            text_content = f.read().strip()
        texts.append(text_content)

        img = load_img(img_path, target_size=(img_height, img_width))
        img_array = img_to_array(img)
        images.append(img_array)

        labels.append(label)

    images = np.array(images, dtype="float32") / 255.0
    labels = np.array(labels)

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<UNK>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    x_texts = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

    return x_texts, images, labels, tokenizer


def load_and_preprocess_test_data(
        test_file="人工智能实验5/src/test_without_label.txt",
        txt_dir="人工智能实验5/src/data/",
        img_dir="人工智能实验5/src/data/",
        tokenizer=None,
        max_len=50,
        img_height=224,
        img_width=224
):
    df_test = pd.read_csv(test_file, sep=",", header=0, names=["guid", "tag"], dtype={"guid": str})

    test_texts = []
    test_images = []
    guid_list = []

    for idx, row in df_test.iterrows():
        guid = str(row["guid"])

        txt_path = os.path.join(txt_dir, guid + ".txt")
        img_path = os.path.join(img_dir, guid + ".jpg")

        if not os.path.isfile(txt_path):
            print(f"[Warning] 测试文本文件不存在: {txt_path}, 跳过该样本")
            continue
        if not os.path.isfile(img_path):
            print(f"[Warning] 测试图片文件不存在: {img_path}, 跳过该样本")
            continue

        with open(txt_path, "r", encoding="ascii", errors="replace") as f:
            text_content = f.read().strip()
        test_texts.append(text_content)

        img = load_img(img_path, target_size=(img_height, img_width))
        img_array = img_to_array(img)
        test_images.append(img_array)

        guid_list.append(guid)

    test_sequences = tokenizer.texts_to_sequences(test_texts)
    x_texts_test = pad_sequences(test_sequences, maxlen=max_len, padding="post", truncating="post")

    x_images_test = np.array(test_images, dtype="float32") / 255.0

    return x_texts_test, x_images_test, guid_list


def build_image_model(img_height=224, img_width=224, num_classes=3):

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.05),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    x_texts, x_images, y_labels, tokenizer = load_and_preprocess_data(
        txt_dir="人工智能实验5/src/data/",
        img_dir="人工智能实验5/src/data/",
        train_file="人工智能实验5/src/train.txt",
        vocab_size=5000,
        max_len=50,
        img_height=224,
        img_width=224
    )

    print("文本输入形状:", x_texts.shape)
    print("图像输入形状:", x_images.shape)
    print("标签形状:", y_labels.shape)

    X_text_train, X_text_val, X_img_train, X_img_val, y_train, y_val = train_test_split(
        x_texts, x_images, y_labels, test_size=0.1, random_state=42, shuffle=True
    )

    y_train_img = to_categorical(y_train, num_classes=3)
    y_val_img = to_categorical(y_val, num_classes=3)

    print("训练集大小:", X_text_train.shape[0])
    print("验证集大小:", X_text_val.shape[0])

    image_model = build_image_model(
        img_height=224,
        img_width=224,
        num_classes=3
    )

    print("\n==== 图像模型结构 ====")
    image_model.summary()

    print("\n==== 训练图像模型 ====")
    image_model.fit(
        X_img_train, y_train_img,
        validation_data=(X_img_val, y_val_img),
        epochs=3,
        batch_size=16,
        verbose=1
    )

    image_probs_val = image_model.predict(X_img_val)

    final_pred_val = np.argmax(image_probs_val, axis=1)

    correct = np.sum(final_pred_val == y_val)
    total = len(y_val)
    acc = correct / total
    print(f"\n图像模型在验证集上的准确率: {acc:.4f}")

    x_texts_test, x_images_test, guid_list = load_and_preprocess_test_data(
        test_file="人工智能实验5/src/test_without_label.txt",
        txt_dir="人工智能实验5/src/data/",
        img_dir="人工智能实验5/src/data/",
        tokenizer=tokenizer,
        max_len=50,
        img_height=224,
        img_width=224
    )

    image_probs_test = image_model.predict(x_images_test)

    final_pred_test = np.argmax(image_probs_test, axis=1)

    idx2label = {0: "negative", 1: "neutral", 2: "positive"}

    pred_labels = [idx2label[idx] for idx in final_pred_test]
    df_result = pd.DataFrame({
        "guid": guid_list,
        "tag": pred_labels
    })

    out_file = "人工智能实验5/src/test_predicted_image_only.txt"
    df_result.to_csv(out_file, index=False, header=True)
    print(f"\n测试集预测完成，结果已保存到: {out_file}")


if __name__ == "__main__":
    main()
