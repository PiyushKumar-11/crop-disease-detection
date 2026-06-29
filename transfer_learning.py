                                                          
                                                         
                                                         
                                                         
                                                          

                                               
                                               
                                               


import os, json
from pathlib import Path

SPLIT_DIR  = "crop_disease_split"
OUTPUT_DIR = "crop_disease_outputs"

                              
required = [
    f"{OUTPUT_DIR}/custom_cnn.keras",
    f"{OUTPUT_DIR}/class_names.json",
    f"{OUTPUT_DIR}/results_summary.json",
]
print("Checking Phase 2 outputs...")
all_ok = True
for f in required:
    exists = os.path.exists(f)
    print(f"  {'✓' if exists else '✗'} {f.split('/')[-1]}")
    if not exists:
        all_ok = False

if not all_ok:
    print("\n⚠ Some Phase 2 files missing — but we can still run Phase 3.")
    print("  Phase 3 only needs the split folder and class_names.json.")
else:
    print("\n✓ All Phase 2 outputs found — ready for Phase 3!")

                  
with open(f"{OUTPUT_DIR}/class_names.json") as f:
    CLASS_NAMES = json.load(f)
N_CLASSES = len(CLASS_NAMES)
print(f"✓ {N_CLASSES} classes loaded")


                                               
                              
                                               
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix
import gc

                                               
                        
                                               

IMG_SIZE   = (128, 128)                                  
BATCH_SIZE = 16                                    

                           
EPOCHS_HEAD = 8                                                                
EPOCHS_FINE = 20                                                                
LR_HEAD     = 1e-3                                         
LR_FINE     = 1e-5                                                          

UNFREEZE_LAYERS = 30                                                      

print(f"Image size      : {IMG_SIZE}")
print(f"Batch size      : {BATCH_SIZE}")
print(f"Head LR         : {LR_HEAD}")
print(f"Fine-tune LR    : {LR_FINE}")
print(f"Layers unfrozen : last {UNFREEZE_LAYERS} of MobileNetV2")


                                               
                                  
                                               

def build_pipelines():
    AUTOTUNE = tf.data.AUTOTUNE

                                    
    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.10),
        layers.RandomBrightness(0.10),
        layers.RandomContrast(0.10),
    ], name="augmentation")

    def normalize(images, labels):
        return tf.cast(images, tf.float32) / 255.0, labels

    def augment_normalize(images, labels):
        return tf.cast(augment(images, training=True), tf.float32) / 255.0, labels

    train_ds = (
        tf.keras.utils.image_dataset_from_directory(
            f"{SPLIT_DIR}/train",
            image_size=IMG_SIZE, batch_size=BATCH_SIZE,
            label_mode="categorical", shuffle=True, seed=42,
        )
        .map(augment_normalize, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)                                 
    )

    val_ds = (
        tf.keras.utils.image_dataset_from_directory(
            f"{SPLIT_DIR}/val",
            image_size=IMG_SIZE, batch_size=BATCH_SIZE,
            label_mode="categorical", shuffle=False,
        )
        .map(normalize, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    test_ds = (
        tf.keras.utils.image_dataset_from_directory(
            f"{SPLIT_DIR}/test",
            image_size=IMG_SIZE, batch_size=BATCH_SIZE,
            label_mode="categorical", shuffle=False,
        )
        .map(normalize, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    print("✓ Train pipeline ready")
    print("✓ Val pipeline ready")
    print("✓ Test pipeline ready")
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = build_pipelines()


                                               
                                  
                                               

def build_mobilenetv2(n_classes, input_shape):
    """
    MobileNetV2 pretrained on ImageNet.
    Custom head added for 38-class plant disease classification.

    Architecture:
      MobileNetV2 base (frozen) → GlobalAvgPool → Dense(512)
      → BatchNorm → Dropout(0.4) → Dense(256) → Dropout(0.3)
      → Dense(38, softmax)
    """
                                                                
    base = MobileNetV2(
        include_top=False,                                               
        weights="imagenet",                                 
        input_shape=input_shape,
    )
    base.trainable = False                                         
    total_layers = len(base.layers)
    print(f"  MobileNetV2 base layers : {total_layers}")
    print(f"  Frozen for Stage 1      : all {total_layers} layers")

                                
    inputs  = layers.Input(shape=input_shape, name="input_layer")
    x       = base(inputs, training=False)                                     
    x       = layers.GlobalAveragePooling2D(name="gap")(x)
    x       = layers.Dense(512, activation="relu", name="dense_512")(x)
    x       = layers.BatchNormalization(name="bn_head")(x)
    x       = layers.Dropout(0.40, name="dropout_1")(x)
    x       = layers.Dense(256, activation="relu", name="dense_256")(x)
    x       = layers.Dropout(0.30, name="dropout_2")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="MobileNetV2_CropDisease")

    trainable   = sum(1 for l in model.layers if l.trainable)
    untrainable = sum(1 for l in model.layers if not l.trainable)
    total_params = model.count_params()

    print(f"  Trainable layers   : {trainable}")
    print(f"  Frozen layers      : {untrainable}")
    print(f"  Total params       : {total_params:,}")
    print(f"  Trainable params   : {sum(np.prod(v.shape) for v in model.trainable_variables):,}")

    return model, base

print("\nBuilding MobileNetV2 model...")
mob_model, mob_base = build_mobilenetv2(N_CLASSES, (*IMG_SIZE, 3))


                                               
                           
                                               

def get_callbacks(name, patience_es=8, patience_lr=4):
    return [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=patience_es,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=patience_lr,
            factor=0.3,
            min_lr=1e-8,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            f"{OUTPUT_DIR}/{name}.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]


                                               
                                   
                                               

print("\n" + "="*55)
print("  STAGE 1: HEAD TRAINING")
print("  Base frozen | Only custom head trains")
print("="*55 + "\n")

mob_model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_HEAD),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
    ]
)

history_stage1 = mob_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    callbacks=get_callbacks("mobilenet_stage1", patience_es=4, patience_lr=2),
    verbose=1,
)

best_s1 = max(history_stage1.history["val_accuracy"])
print(f"\n  ✅ Stage 1 complete!")
print(f"     Best val accuracy : {best_s1*100:.2f}%")
print(f"     (Head learned to classify diseases using frozen ImageNet features)")


                                               
                                        
                                               

print("\n" + "="*55)
print(f"  STAGE 2: FINE-TUNING (top {UNFREEZE_LAYERS} layers)")
print("  Unfreeze top MobileNetV2 layers + retrain at low LR")
print("="*55 + "\n")

                               
mob_base.trainable = True
for layer in mob_base.layers[:-UNFREEZE_LAYERS]:
    layer.trainable = False

                            
trainable_now = sum(1 for l in mob_base.layers if l.trainable)
print(f"  MobileNetV2 trainable layers : {trainable_now} / {len(mob_base.layers)}")
print(f"  Fine-tune learning rate      : {LR_FINE} (very small to avoid destroying pretrained weights)\n")

                              
mob_model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_FINE),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
    ]
)

history_stage2 = mob_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE,
    callbacks=get_callbacks("mobilenet_final", patience_es=7, patience_lr=3),
    verbose=1,
)

best_s2 = max(history_stage2.history["val_accuracy"])
print(f"\n  ✅ Stage 2 complete!")
print(f"     Best val accuracy : {best_s2*100:.2f}%")
print(f"     Improvement over Stage 1 : +{(best_s2 - best_s1)*100:.2f}%")


                                               
                                
                                               

print("\nEvaluating on test set...")
mob_results = mob_model.evaluate(test_ds, verbose=0)
test_acc  = mob_results[1] * 100
test_top3 = mob_results[2] * 100

print(f"  Test Loss      : {mob_results[0]:.4f}")
print(f"  Test Accuracy  : {test_acc:.2f}%")
print(f"  Top-3 Accuracy : {test_top3:.2f}%")

                       
def _to_class_indices(values):
    array = values.numpy() if hasattr(values, "numpy") else np.asarray(values)
    array = np.asarray(array)

    if array.ndim == 0:
        return np.array([int(array)], dtype=int)
    if array.ndim == 1:
        return array.astype(int)
    if array.shape[-1] > 1:
        return np.argmax(array, axis=-1).astype(int)
    return array.reshape(-1).astype(int)


y_pred_list, y_true_list = [], []
for images, labels in test_ds:
    preds = mob_model.predict(images, verbose=0)
    y_pred_list.append(np.argmax(preds, axis=1).astype(int))
    y_true_list.append(_to_class_indices(labels))

y_pred = np.concatenate(y_pred_list).astype(int)
y_true = np.concatenate(y_true_list).astype(int)

if y_true.shape[0] != y_pred.shape[0]:
    raise ValueError(f"Prediction/label count mismatch: {y_true.shape[0]} vs {y_pred.shape[0]}")

                            
report = classification_report(
    y_true, y_pred,
    target_names=[c.replace("___"," - ").replace("_"," ") for c in CLASS_NAMES],
    zero_division=0
)
print("\nPer-class Report:")
print(report)

with open(f"{OUTPUT_DIR}/mobilenet_classification_report.txt", "w") as f:
    f.write(f"MobileNetV2 Transfer Learning\n")
    f.write(f"Test Accuracy  : {test_acc:.2f}%\n")
    f.write(f"Top-3 Accuracy : {test_top3:.2f}%\n\n")
    f.write(report)
print("✓ Classification report saved")


                                               
                  
                                               

                                                     
s1_len = len(history_stage1.history["accuracy"])
combined = {
    "acc":   history_stage1.history["accuracy"]     + history_stage2.history["accuracy"],
    "vacc":  history_stage1.history["val_accuracy"] + history_stage2.history["val_accuracy"],
    "loss":  history_stage1.history["loss"]         + history_stage2.history["loss"],
    "vloss": history_stage1.history["val_loss"]     + history_stage2.history["val_loss"],
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("MobileNetV2 — Training History (Stage 1 + Fine-tuning)",
             fontsize=13, fontweight="bold")

for ax, tm, vm, title in [
    (axes[0], combined["acc"],  combined["vacc"],  "Accuracy"),
    (axes[1], combined["loss"], combined["vloss"], "Loss"),
]:
    epochs_range = range(len(tm))
    ax.plot(epochs_range, tm, label="Train",      color="#1D9E75", lw=2)
    ax.plot(epochs_range, vm, label="Validation", color="#E05C2A", lw=2)
    ax.axvline(x=s1_len - 1, color="#4A90D9", ls="--", lw=1.5,
               alpha=0.8, label=f"Fine-tune starts (epoch {s1_len})")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/phase3_training_history.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Training history saved")


                                               
                                               
                                               

                  
mob_model.save(f"{OUTPUT_DIR}/mobilenetv2_final.keras")
print(f"✓ Model saved: mobilenetv2_final.keras")

                                             
try:
    with open(f"{OUTPUT_DIR}/results_summary.json") as f:
        summary = json.load(f)
except:
    summary = {}

summary.update({
    "mobilenetv2_accuracy":      round(test_acc,  2),
    "mobilenetv2_top3_accuracy": round(test_top3, 2),
    "mobilenetv2_stage1_best":   round(best_s1 * 100, 2),
    "mobilenetv2_stage2_best":   round(best_s2 * 100, 2),
    "img_size":                  list(IMG_SIZE),
    "n_classes":                 N_CLASSES,
})

with open(f"{OUTPUT_DIR}/results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("✓ Results summary updated")

               
print("\n" + "="*55)
print("  ✅ PHASE 3 COMPLETE!")
print("="*55)
print(f"\n  Stage 1 (head only)  best val : {best_s1*100:.2f}%")
print(f"  Stage 2 (fine-tuned) best val : {best_s2*100:.2f}%")
print(f"  Final test accuracy            : {test_acc:.2f}%")
print(f"  Final top-3 accuracy           : {test_top3:.2f}%")
if cnn_acc:
    print(f"\n  Custom CNN (Phase 2)  : {cnn_acc:.2f}%")
    print(f"  MobileNetV2 (Phase 3) : {test_acc:.2f}%")
    print(f"  Improvement           : +{test_acc - cnn_acc:.2f}% from transfer learning ⭐")
print(f"\n  All outputs saved to:")
print(f"  {OUTPUT_DIR}/")
print("    • mobilenetv2_final.keras")
print("    • phase3_training_history.png")
print("    • phase3_confusion_matrix.png")
print("    • phase3_model_comparison.png")
print("    • phase3_sample_predictions.png")
print("    • mobilenet_classification_report.txt")
print("    • results_summary.json")
print("\n  Next → phase4_app.py (Streamlit demo)")
print("="*55)