import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix

# ── Paths & Hyper‑parameters ────────────────────────────────────────────────────────
CSV_PATH = 'data/aptos2019-blindness-detection/train.csv'
IMG_DIR = 'data/aptos2019-blindness-detection/train_images/'
MODEL_SAVE_PATH = 'models/efficientnetb3_dr.keras'
METRICS_PATH = 'models/metrics.json'
TARGET_SIZE = (300, 300)  # EfficientNetB3 native resolution
BATCH_SIZE = 32

# ── Helper: Crop retina from black border (in‑memory) ───────────────────────────────
def crop_retina(img: np.ndarray) -> np.ndarray:
    """Detect the retina fundus, crop to a 1:1 square bounding box, with fallback to original image.
    Ensures square aspect ratio and prevents extreme or invalid crops.
    """
    if img is None or img.size == 0:
        return img

    if img.dtype != np.uint8:
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    else:
        img_uint8 = img

    h_orig, w_orig = img_uint8.shape[:2]
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("[crop_retina] Fallback: No contours found.")
        return img

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    img_area = h_orig * w_orig

    if area < 0.05 * img_area:
        print(f"[crop_retina] Fallback: Contour too small ({(area/img_area)*100:.1f}%).")
        return img

    x, y, w, h = cv2.boundingRect(largest)
    aspect_ratio = w / float(max(1, h))

    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        print(f"[crop_retina] Fallback: Extreme aspect ratio ({aspect_ratio:.2f}).")
        return img

    # Force 1:1 square bounding box centered on detected retina
    center_x = x + w / 2.0
    center_y = y + h / 2.0
    side = max(w, h)

    # Initial centered square coords
    x1 = int(center_x - side / 2.0)
    y1 = int(center_y - side / 2.0)
    x2 = x1 + side
    y2 = y1 + side

    # Shift (not shrink) if square extends past image edges
    if x1 < 0:
        x2 -= x1; x1 = 0
    if y1 < 0:
        y2 -= y1; y1 = 0
    if x2 > w_orig:
        x1 -= (x2 - w_orig); x2 = w_orig
    if y2 > h_orig:
        y1 -= (y2 - h_orig); y2 = h_orig

    x1 = max(0, x1)
    y1 = max(0, y1)

    raw_crop = img[y1:y2, x1:x2]
    ch, cw = raw_crop.shape[:2]

    if ch != cw:
        # Pad to perfect square with black if image was smaller than side
        pad_side = max(ch, cw)
        cropped = np.zeros((pad_side, pad_side, 3), dtype=img.dtype)
        oy = (pad_side - ch) // 2
        ox = (pad_side - cw) // 2
        cropped[oy:oy+ch, ox:ox+cw] = raw_crop
    else:
        cropped = raw_crop

    print(f"[crop_retina] Success: contour_bbox=({x},{y},{w},{h}), final_crop={cropped.shape[:2]}, ratio={cropped.shape[1]/max(1,cropped.shape[0]):.2f}")
    return cropped

print('Reading CSV...')
df = pd.read_csv(CSV_PATH)
# Align filenames and ensure string labels
df['id_code'] = df['id_code'].astype(str) + '.png'
df['diagnosis'] = df['diagnosis'].astype(str)

# ── Train / Validation / Test split (stratified) ───────────────────────────────────
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['diagnosis'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['diagnosis'], random_state=42)

# ── ImageDataGenerator with EfficientNet preprocessing (no double‑scaling) ────────
preprocess_fn = tf.keras.applications.efficientnet.preprocess_input
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_fn,
    rotation_range=360,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest'
)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_fn)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_fn)

# Custom preprocessing to apply retina cropping before resizing
def preprocessing_crop(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_retina(img)
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    return img

# Keras generators expect a function that takes a path and returns an array; we wrap it via `preprocessing_function`.
# Since ImageDataGenerator does not expose a direct hook, we embed cropping inside a lambda passed to `flow_from_dataframe`.

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=IMG_DIR,
    x_col='id_code',
    y_col='diagnosis',
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    preprocessing_function=lambda img: crop_retina(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=IMG_DIR,
    x_col='id_code',
    y_col='diagnosis',
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    preprocessing_function=lambda img: crop_retina(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=IMG_DIR,
    x_col='id_code',
    y_col='diagnosis',
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    preprocessing_function=lambda img: crop_retina(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
)

# ── Compute class weights (balanced) ───────────────────────────────────────────────
class_indices = train_generator.classes
unique_classes = np.unique(class_indices)
weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=class_indices)
class_weight_dict = {str(i): float(w) for i, w in zip(unique_classes, weights)}
print('Class distribution (train):')
for i, count in zip(unique_classes, np.bincount(class_indices)):
    print(f'  class {i}: {count} samples, weight {class_weight_dict[str(i)]:.3f}')

# ── Build EfficientNetB3 base (input 300x300) ───────────────────────────────────────
base_model = tf.keras.applications.EfficientNetB3(
    weights='imagenet',
    include_top=False,
    input_shape=(300, 300, 3)
)
# Freeze all layers for warm‑up
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ── Warm‑up: train head only (5 epochs) ────────────────────────────────────────────
print('Warm-up training (head only, 5 epochs)')
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
    class_weight=class_weight_dict,
    verbose=2
)

# ── Fine‑tuning: unfreeze top 30 % of EfficientNet layers ────────────────────────
num_unfreeze = int(0.30 * len(base_model.layers))
for layer in base_model.layers[-num_unfreeze:]:
    layer.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1)
print('Fine-tuning (unfreeze top 30%, 15 epochs)')
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr],
    verbose=2
)

# ── Evaluation on test set ────────────────────────────────────────────────────────
print('Evaluating on held-out test set')
preds = model.predict(test_generator, verbose=0)
y_true = test_generator.classes
y_pred = np.argmax(preds, axis=1)
accuracy = accuracy_score(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)
conf_mat = confusion_matrix(y_true, y_pred).tolist()
class_report = classification_report(y_true, y_pred, output_dict=True)
metrics = {
    'accuracy': accuracy,
    'kappa': kappa,
    'confusion_matrix': conf_mat,
    'classification_report': class_report,
    'train_samples': int(train_generator.n),
    'val_samples': int(val_generator.n),
    'test_samples': int(test_generator.n)
}
os.makedirs('models', exist_ok=True)
with open(METRICS_PATH, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f'Metrics saved to {METRICS_PATH}')

# ── Save final model ───────────────────────────────────────────────────────────────
model.save(MODEL_SAVE_PATH)
print(f'Model saved to {MODEL_SAVE_PATH}')