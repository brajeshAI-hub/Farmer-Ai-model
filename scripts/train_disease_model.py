import json
from pathlib import Path

import tensorflow as tf


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "plant_disease"
    model_dir = project_root / "models"
    model_dir.mkdir(exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(
            "Missing dataset directory at data/plant_disease. "
            "Use class folders (e.g., Healthy/, Leaf_Blight/, Rust/)."
        )

    image_size = (224, 224)
    batch_size = 32

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
    )

    class_names = train_ds.class_names

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1.0 / 255),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(class_names), activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)
    final_val_acc = history.history["val_accuracy"][-1]
    print(f"Final validation accuracy: {final_val_acc:.4f}")

    model_path = model_dir / "plant_disease_mobilenet.keras"
    labels_path = model_dir / "disease_labels.json"

    model.save(model_path)
    labels_path.write_text(json.dumps(class_names, indent=2))

    print(f"Saved disease model to {model_path}")
    print(f"Saved labels to {labels_path}")


if __name__ == "__main__":
    main()
