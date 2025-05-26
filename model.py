import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# --- 1. Kiểm tra thư mục dữ liệu ---
train_dir = "data/chest_xray/train"
val_dir = "data/chest_xray/val"
test_dir = "data/chest_xray/test"

def check_directory(dir_path, dir_name):
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Thư mục {dir_name} không tồn tại: {dir_path}")
    if not os.listdir(dir_path):
        raise ValueError(f"Thư mục {dir_name} trống: {dir_path}")

for dir_path, dir_name in [(train_dir, "train"), (val_dir, "val"), (test_dir, "test")]:
    check_directory(dir_path, dir_name)

# --- 2. Tạo dataset với tf.data ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

try:
    val_samples = sum(len(os.listdir(os.path.join(val_dir, cls))) 
                     for cls in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, cls)))
    print(f"Số mẫu trong val_dir: {val_samples}")

    if val_samples < 32:
        print("⚠️ Tập validation quá nhỏ, sử dụng 20% tập train làm validation.")
        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=True,
            label_mode='int',
            validation_split=0.2,
            subset='training',
            seed=42
        )
        val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=True,
            label_mode='int',
            validation_split=0.2,
            subset='validation',
            seed=42
        )
    else:
        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=True,
            label_mode='int',
            seed=42
        )
        val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            val_dir,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=True,
            label_mode='int',
            seed=42
        )

    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode='int'
    )

except Exception as e:
    raise RuntimeError(f"Lỗi khi tải dữ liệu: {str(e)}")

class_names = train_dataset.class_names
train_labels = np.concatenate([y.numpy() for _, y in train_dataset], axis=0)
val_labels = np.concatenate([y.numpy() for _, y in val_dataset], axis=0)
test_labels = np.concatenate([y.numpy() for _, y in test_dataset], axis=0)

print(f"Phân bố lớp trong tập huấn luyện: {np.bincount(train_labels)}")
print(f"Số mẫu huấn luyện: {len(train_labels)}")
print(f"Số mẫu kiểm định: {len(val_labels)}")
print(f"Số mẫu kiểm tra: {len(test_labels)}")

# --- 3. Data Augmentation ---
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomShear(0.15)
])

# Chuẩn hóa và augmentation
train_dataset = train_dataset.map(
    lambda x, y: (data_augmentation(x, training=True) / 255.0, y),
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

val_dataset = val_dataset.map(
    lambda x, y: (x / 255.0, y),
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.map(
    lambda x, y: (x / 255.0, y),
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

# --- 4. Xử lý dữ liệu không cân bằng ---
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weight_dict = dict(enumerate(class_weights))
print(f"Trọng số lớp: {class_weight_dict}")

# --- 5. Xây dựng mô hình ---
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# --- 6. Compile ---
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# --- 7. Callback ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint("best_densenet121_model.keras", monitor='val_loss', save_best_only=True, verbose=1)
]

# --- 8. Huấn luyện lần 1 ---
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# --- 9. Fine-tuning ---
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

history_fine = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# --- 10. Đánh giá ---
loss, acc, prec, rec = model.evaluate(test_dataset)
print(f"✅ Test Accuracy: {acc:.4f}")
print(f"✅ Test Precision: {prec:.4f}")
print(f"✅ Test Recall: {rec:.4f}")

# --- 11. Báo cáo phân loại ---
y_pred_prob = model.predict(test_dataset).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)
print("📊 Classification Report:")
print(classification_report(test_labels, y_pred, target_names=class_names))

# --- 12. Lưu mô hình ---
model.save("final_densenet121_pneumonia_model.keras")
print("💾 Mô hình đã được lưu!")

# --- 13. Vẽ biểu đồ ---
def plot_metric(histories, metric, title, subplot_pos):
    values = [h.history[metric] for h in histories if metric in h.history]
    val_values = [h.history[f'val_{metric}'] for h in histories if f'val_{metric}' in h.history]
    combined = sum(values, [])
    combined_val = sum(val_values, [])
    plt.subplot(2, 2, subplot_pos)
    plt.plot(combined, label='Train')
    plt.plot(combined_val, label='Val')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)

plt.figure(figsize=(12, 8))
plot_metric([history, history_fine], 'accuracy', 'Accuracy', 1)
plot_metric([history, history_fine], 'loss', 'Loss', 2)
plot_metric([history, history_fine], 'precision', 'Precision', 3)
plot_metric([history, history_fine], 'recall', 'Recall', 4)
plt.tight_layout()
plt.show()
