"""
Crop Disease Detection using CNN
Phase 2: Custom CNN from Scratch
-----------------------------------------
Architecture:
  Block 1: Conv(32)  → BN → ReLU → MaxPool → Dropout
  Block 2: Conv(64)  → BN → ReLU → MaxPool → Dropout
  Block 3: Conv(128) → BN → ReLU → MaxPool → Dropout
  Block 4: Conv(256) → BN → ReLU → MaxPool → Dropout
  Head: GlobalAvgPool → Dense(512) → Dropout → Dense(38, softmax)

Expected accuracy: 70–80% (good baseline before transfer learning)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

                                               
               
                                               
SPLIT_DIR   = "crop_disease_split"
OUTPUT_DIR  = "crop_disease_outputs"
META_PATH   = f"{OUTPUT_DIR}/metadata.json"
MODEL_PATH  = f"{OUTPUT_DIR}/custom_cnn.keras"

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 1e-3

os.makedirs(OUTPUT_DIR, exist_ok=True)

                                               
                                   
                                               
def load_pipelines():
    with open(META_PATH) as f:
        meta = json.load(f)

    class_names = meta["class_names"]
    n_classes   = meta["n_classes"]
    AUTOTUNE    = tf.data.AUTOTUNE

    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.10),
        layers.RandomBrightness(0.10),
        layers.RandomContrast(0.10),
    ])

    def normalize(images, labels):
        return tf.cast(images, tf.float32) / 255.0, labels

    def augment_and_normalize(images, labels):
        images = augment(images, training=True)
        return tf.cast(images, tf.float32) / 255.0, labels

    train_ds = (tf.keras.utils.image_dataset_from_directory(
                    f"{SPLIT_DIR}/train", image_size=IMG_SIZE,
                    batch_size=BATCH_SIZE, label_mode="categorical",
                    shuffle=True, seed=42)
                .map(augment_and_normalize, num_parallel_calls=AUTOTUNE)
                .cache().prefetch(AUTOTUNE))

    val_ds   = (tf.keras.utils.image_dataset_from_directory(
                    f"{SPLIT_DIR}/val", image_size=IMG_SIZE,
                    batch_size=BATCH_SIZE, label_mode="categorical", shuffle=False)
                .map(normalize, num_parallel_calls=AUTOTUNE)
                .cache().prefetch(AUTOTUNE))

    test_ds  = (tf.keras.utils.image_dataset_from_directory(
                    f"{SPLIT_DIR}/test", image_size=IMG_SIZE,
                    batch_size=BATCH_SIZE, label_mode="categorical", shuffle=False)
                .map(normalize, num_parallel_calls=AUTOTUNE)
                .cache().prefetch(AUTOTUNE))

    print(f"  ✓ Pipelines loaded | {n_classes} classes | {len(class_names)} labels")
    return train_ds, val_ds, test_ds, class_names, n_classes


                                               
                          
                                               
def build_custom_cnn(n_classes, input_shape=(224, 224, 3)):
    """
    4-block CNN with BatchNorm and GlobalAveragePooling.
    BatchNorm helps stabilise training across 38 classes.
    GlobalAvgPool instead of Flatten reduces overfitting.
    """
    inputs = layers.Input(shape=input_shape, name="input")
    x = inputs

                                
    for filters, pool in [(32, True), (64, True), (128, True), (256, True)]:
        x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        if pool:
            x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

                               
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.40)(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="CustomCNN")
    return model


                                               
               
                                               
def train_model(model, train_ds, val_ds):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")]
    )

    model.summary()
    total_params = model.count_params()
    print(f"\n  Total parameters: {total_params:,}")

    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=4,
            factor=0.3, min_lr=1e-6, verbose=1
        ),
        callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
    ]

    print(f"\nTraining Custom CNN for up to {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=cb_list,
        verbose=1,
    )
    return history


                                               
                  
                                               
def evaluate_model(model, test_ds, class_names):
    print("\nEvaluating on test set...")
    results = model.evaluate(test_ds, verbose=0)
    print(f"  Test Loss     : {results[0]:.4f}")
    print(f"  Test Accuracy : {results[1]*100:.2f}%")
    print(f"  Top-3 Accuracy: {results[2]*100:.2f}%")

                     
    y_pred, y_true = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

                                        
    short_names = [c.replace("___", "\n").replace("_", " ") for c in class_names]

                      
    report = classification_report(
        y_true, y_pred,
        target_names=[c.replace("___", " - ").replace("_", " ") for c in class_names],
        zero_division=0
    )
    print("\nPer-class Classification Report:")
    print(report)

                 
    with open(f"{OUTPUT_DIR}/custom_cnn_classification_report.txt", "w") as f:
        f.write(f"Custom CNN Results\n")
        f.write(f"Test Accuracy: {results[1]*100:.2f}%\n")
        f.write(f"Top-3 Accuracy: {results[2]*100:.2f}%\n\n")
        f.write(report)

    return y_true, y_pred, results


                                               
               
                                               
def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Custom CNN — Training History", fontsize=13, fontweight="bold")

    axes[0].plot(history.history["accuracy"],     label="Train",      color="#1D9E75", lw=2)
    axes[0].plot(history.history["val_accuracy"], label="Validation", color="#E05C2A", lw=2)
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Train",      color="#1D9E75", lw=2)
    axes[1].plot(history.history["val_loss"], label="Validation", color="#E05C2A", lw=2)
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_custom_cnn_training.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: 04_custom_cnn_training.png")


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
                                   
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    short = [c.split("___")[1].replace("_", " ")[:12] for c in class_names]
    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(cm_norm, annot=False, cmap="Greens",
                xticklabels=short, yticklabels=short,
                ax=ax, linewidths=0.1)
    ax.set_title("Custom CNN — Confusion Matrix (% per row)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0,  fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_custom_cnn_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: 05_custom_cnn_confusion_matrix.png")


def plot_sample_predictions(model, test_ds, class_names, n=12):
    images_batch, labels_batch = next(iter(test_ds))
    preds = model.predict(images_batch[:n], verbose=0)

    short = [c.replace("___", "\n").replace("_", " ") for c in class_names]
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    fig.suptitle("Custom CNN — Sample Predictions", fontsize=13, fontweight="bold")

    for i, ax in enumerate(axes.flatten()):
        img   = images_batch[i].numpy()
        true  = np.argmax(labels_batch[i].numpy())
        pred  = np.argmax(preds[i])
        conf  = preds[i][pred] * 100
        correct = true == pred

        ax.imshow(np.clip(img, 0, 1))
        color = "#1D9E75" if correct else "#E05C2A"
        mark  = "✓" if correct else "✗"
        ax.set_title(
            f"{mark} True: {short[true]}\nPred: {short[pred]} ({conf:.0f}%)",
            fontsize=7.5, color=color, fontweight="bold"
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/06_custom_cnn_predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: 06_custom_cnn_predictions.png")


                                               
      
                                               
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  CROP DISEASE DETECTION — PHASE 2")
    print("  Custom CNN from Scratch")
    print("=" * 55 + "\n")

                  
    train_ds, val_ds, test_ds, class_names, n_classes = load_pipelines()

                    
    print("\nBuilding Custom CNN architecture...")
    model = build_custom_cnn(n_classes)

              
    history = train_model(model, train_ds, val_ds)

                 
    y_true, y_pred, results = evaluate_model(model, test_ds, class_names)

              
    print("\nGenerating charts...")
    plot_training_history(history)
    plot_confusion_matrix(y_true, y_pred, class_names)
    plot_sample_predictions(model, test_ds, class_names)

    print("\n" + "=" * 55)
    print("  ✅ Phase 2 Complete!")
    print(f"  Test Accuracy : {results[1]*100:.2f}%")
    print(f"  Top-3 Accuracy: {results[2]*100:.2f}%")
    print(f"  Model saved   : {MODEL_PATH}")
    print("  Next → python phase3_transfer_learning.py")
    print("=" * 55 + "\n")