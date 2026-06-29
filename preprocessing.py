"""
Crop Disease Detection using CNN
Phase 1: Dataset Setup & Preprocessing
-----------------------------------------
Dataset : PlantVillage (38 classes, ~54,000 images)
          Download from Kaggle:
          https://www.kaggle.com/datasets/emmarex/plantdisease

Folder structure expected after download:
  PlantVillage/
    Apple___Apple_scab/
    Apple___Black_rot/
    Apple___Cedar_apple_rust/
    Apple___healthy/
    Corn_(maize)___Common_rust/
    ...
    Tomato___Late_blight/
    Tomato___healthy/
    (38 folders total)

This script:
  1. Analyses the dataset (class distribution, sample counts)
  2. Splits into train / val / test (70/15/15)
  3. Builds tf.data pipelines with augmentation
  4. Saves class names and split info
"""

import os
import json
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

                                               
               
                                               
DATASET_DIR  = "PlantVillage"                                         
OUTPUT_DIR   = "crop_disease_outputs"
SPLIT_DIR    = "crop_disease_split"                                     

IMG_SIZE     = (224, 224)                                                        
BATCH_SIZE   = 32
RANDOM_SEED  = 42

TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15                                     

os.makedirs(OUTPUT_DIR, exist_ok=True)

                                               
                         
                                               
def analyse_dataset(dataset_dir):
    print("Analysing PlantVillage dataset...")
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        print(f"\n  ⚠ Dataset folder '{dataset_dir}' not found!")
        print("  Please download it from Kaggle:")
        print("  https://www.kaggle.com/datasets/emmarex/plantdisease")
        print("  Then extract and place the 'PlantVillage' folder here.\n")
        return None, None

    class_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]

    print(f"  ✓ Found {len(class_names)} classes")

    class_counts = {}
    total = 0
    for d in class_dirs:
        imgs = list(d.glob("*.jpg")) + list(d.glob("*.JPG")) + \
               list(d.glob("*.jpeg")) + list(d.glob("*.png"))
        class_counts[d.name] = len(imgs)
        total += len(imgs)

    print(f"  ✓ Total images : {total:,}")
    print(f"  ✓ Avg per class: {total // len(class_names)}")
    print(f"  ✓ Min class    : {min(class_counts.values())} ({min(class_counts, key=class_counts.get)})")
    print(f"  ✓ Max class    : {max(class_counts.values())} ({max(class_counts, key=class_counts.get)})")

                    
    plant_groups = defaultdict(list)
    for cls in class_names:
        parts = cls.split("___", 1)
        plant = parts[0].replace("_", " ")
        plant_groups[plant].append(cls)

    print(f"\n  Plants covered ({len(plant_groups)}):")
    for plant, classes in plant_groups.items():
        diseases = []
        for c in classes:
            parts = c.split("___", 1)
            if len(parts) == 2:
                diseases.append(parts[1].replace("_", " "))
            else:
                fallback = c.replace("_", " ")
                diseases.append(fallback)
        print(f"    {plant:<25} → {', '.join(diseases)}")

    return class_names, class_counts


                                               
                                 
                                               
def plot_class_distribution(class_counts):
    labels  = [k.replace("___", "\n").replace("_", " ") for k in class_counts.keys()]
    values  = list(class_counts.values())
    colors  = ["#1D9E75" if "healthy" in k.lower() else "#E05C2A" for k in class_counts.keys()]

    fig, ax = plt.subplots(figsize=(18, 7))
    bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_title("PlantVillage Dataset — Images per Class\n(Green = Healthy, Orange = Disease)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Images")
    ax.grid(True, alpha=0.3, axis="y")

            
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#1D9E75", label="Healthy"),
        Patch(color="#E05C2A", label="Diseased"),
    ], fontsize=10)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/01_class_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ Saved: 01_class_distribution.png")


                                               
                            
                                               
def plot_sample_images(dataset_dir, class_names, n_classes=12):
    dataset_path = Path(dataset_dir)
    sample_classes = random.sample(class_names, min(n_classes, len(class_names)))

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    fig.suptitle("PlantVillage — Sample Images per Class",
                 fontsize=13, fontweight="bold")

    for ax, cls in zip(axes.flatten(), sample_classes):
        cls_path = dataset_path / cls
        imgs = list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.JPG")) + \
               list(cls_path.glob("*.jpeg"))
        if imgs:
            img = mpimg.imread(str(random.choice(imgs)))
            ax.imshow(img)
            label = cls.replace("___", "\n").replace("_", " ")
            color = "#1D9E75" if "healthy" in cls.lower() else "#E05C2A"
            ax.set_title(label, fontsize=8, color=color, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/02_sample_images.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: 02_sample_images.png")


                                               
                                  
                                               
def split_dataset(dataset_dir, class_names, split_dir):
    print(f"\nSplitting dataset into train/val/test ({TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO})...")
    dataset_path = Path(dataset_dir)
    split_path   = Path(split_dir)

                          
    for split in ["train", "val", "test"]:
        for cls in class_names:
            (split_path / split / cls).mkdir(parents=True, exist_ok=True)

    random.seed(RANDOM_SEED)
    split_counts = {"train": 0, "val": 0, "test": 0}

    for cls in class_names:
        cls_path = dataset_path / cls
        images   = list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.JPG")) + \
                   list(cls_path.glob("*.jpeg")) + list(cls_path.glob("*.png"))
        random.shuffle(images)

        n       = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        splits_map = {
            "train": images[:n_train],
            "val":   images[n_train:n_train + n_val],
            "test":  images[n_train + n_val:],
        }

        for split_name, split_imgs in splits_map.items():
            dest_dir = split_path / split_name / cls
            for img_path in split_imgs:
                shutil.copy2(str(img_path), str(dest_dir / img_path.name))
            split_counts[split_name] += len(split_imgs)

    print(f"  ✓ Train : {split_counts['train']:,} images")
    print(f"  ✓ Val   : {split_counts['val']:,} images")
    print(f"  ✓ Test  : {split_counts['test']:,} images")
    return split_counts


                                               
                                 
                                               
def build_data_pipeline(split_dir):
    """
    Builds efficient tf.data pipelines for train, val, test.
    Applies augmentation on training data only.
    """
    print("\nBuilding tf.data pipelines...")
    split_path = Path(split_dir)
    AUTOTUNE   = tf.data.AUTOTUNE

                                                 
    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(split_path / "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
        seed=RANDOM_SEED,
    )

                                                 
    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(split_path / "val"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )

                                           
    test_ds = tf.keras.utils.image_dataset_from_directory(
        str(split_path / "test"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = train_ds.class_names
    n_classes   = len(class_names)
    print(f"  ✓ Classes detected: {n_classes}")

                                   
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomBrightness(0.10),
        tf.keras.layers.RandomContrast(0.10),
    ], name="augmentation")

                                                 
    def normalize(images, labels):
        return tf.cast(images, tf.float32) / 255.0, labels

    def augment_and_normalize(images, labels):
        images = augment(images, training=True)
        return tf.cast(images, tf.float32) / 255.0, labels

    train_ds = (train_ds
                .map(augment_and_normalize, num_parallel_calls=AUTOTUNE)
                .cache()
                .prefetch(AUTOTUNE))

    val_ds   = (val_ds
                .map(normalize, num_parallel_calls=AUTOTUNE)
                .cache()
                .prefetch(AUTOTUNE))

    test_ds  = (test_ds
                .map(normalize, num_parallel_calls=AUTOTUNE)
                .cache()
                .prefetch(AUTOTUNE))

    print(f"  ✓ Augmentation applied to training set")
    print(f"  ✓ All pipelines ready (cache + prefetch enabled)")
    return train_ds, val_ds, test_ds, class_names, n_classes


                                               
                                   
                                               
def plot_augmentation_samples(split_dir, class_names):
    split_path = Path(split_dir)
    cls   = random.choice(class_names)
    imgs  = list((split_path / "train" / cls).glob("*.jpg"))[:1]
    if not imgs:
        return

    img    = tf.keras.utils.load_img(str(imgs[0]), target_size=IMG_SIZE)
    img_arr = tf.keras.utils.img_to_array(img) / 255.0
    img_tensor = tf.expand_dims(img_arr, 0)

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomBrightness(0.10),
    ])

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle(f"Data Augmentation Samples — {cls.replace('___', ' → ').replace('_', ' ')}",
                 fontsize=12, fontweight="bold")

    axes[0, 0].imshow(img_arr)
    axes[0, 0].set_title("Original", fontsize=9)
    axes[0, 0].axis("off")

    for i, ax in enumerate(axes.flatten()[1:]):
        aug_img = augment(img_tensor, training=True)[0].numpy()
        ax.imshow(np.clip(aug_img, 0, 1))
        ax.set_title(f"Aug #{i+1}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/03_augmentation_samples.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: 03_augmentation_samples.png")


                                               
                       
                                               
def save_metadata(class_names, class_counts, split_counts):
    meta = {
        "class_names":  class_names,
        "n_classes":    len(class_names),
        "class_counts": class_counts,
        "split_counts": split_counts,
        "img_size":     list(IMG_SIZE),
        "batch_size":   BATCH_SIZE,
    }
    path = f"{OUTPUT_DIR}/metadata.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✓ Metadata saved: {path}")


                                               
      
                                               
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  CROP DISEASE DETECTION — PHASE 1")
    print("  Dataset Setup & Preprocessing")
    print("=" * 55 + "\n")

                
    class_names, class_counts = analyse_dataset(DATASET_DIR)
    if class_names is None:
        exit(1)

                               
    plot_class_distribution(class_counts)
    plot_sample_images(DATASET_DIR, class_names)

              
    split_counts = split_dataset(DATASET_DIR, class_names, SPLIT_DIR)

                        
    train_ds, val_ds, test_ds, class_names, n_classes = build_data_pipeline(SPLIT_DIR)

                             
    plot_augmentation_samples(SPLIT_DIR, class_names)

                      
    save_metadata(class_names, class_counts, split_counts)

    print("\n" + "=" * 55)
    print("  ✅ Phase 1 Complete!")
    print(f"  Outputs in: {OUTPUT_DIR}/")
    print("    • 01_class_distribution.png")
    print("    • 02_sample_images.png")
    print("    • 03_augmentation_samples.png")
    print("    • metadata.json")
    print(f"  Split data in: {SPLIT_DIR}/")
    print("    • train/  val/  test/")
    print("  Next → python phase2_custom_cnn.py")
    print("=" * 55 + "\n")