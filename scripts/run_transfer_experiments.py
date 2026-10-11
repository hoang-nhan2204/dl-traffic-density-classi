"""Resumable TensorFlow transfer-learning runner; all selection uses validation Macro-F1."""
import csv, gc, json, os, sys, time, traceback
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.transfer_data import create_transfer_sequences

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "transfer_learning"
REGISTRY = OUT / "experiment_registry.csv"
CLASSES = ["Empty", "Low", "Medium", "High", "Traffic Jam"]


def set_seed(seed):
    tf.keras.utils.set_random_seed(seed)


def make_augment(name):
    if name == "A0": return None
    strength = 0.03 if name == "A1" else 0.05
    contrast = 0.10 if name == "A1" else 0.20
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(strength),
        tf.keras.layers.RandomTranslation(strength, strength),
        tf.keras.layers.RandomZoom((-0.05, 0.05) if name == "A1" else (-0.10, 0.10)),
        tf.keras.layers.RandomContrast(contrast),
    ], name=f"{name}_train_augmentation")


def backbone(name, input_tensor):
    apps = tf.keras.applications
    if name == "mobilenetv2":
        return apps.MobileNetV2(include_top=False, weights="imagenet", input_tensor=apps.mobilenet_v2.preprocess_input(input_tensor)), 1e-3
    if name == "efficientnetb0":
        # EfficientNetB0 Keras weights include their Rescaling preprocessing layer.
        return apps.EfficientNetB0(include_top=False, weights="imagenet", input_tensor=input_tensor), 1e-3
    return apps.ResNet50(include_top=False, weights="imagenet", input_tensor=apps.resnet50.preprocess_input(input_tensor)), 1e-3


def build_model(name, augmentation):
    raw = tf.keras.Input((224, 224, 3))
    x = make_augment(augmentation)(raw) if augmentation != "A0" else raw
    base, _ = backbone(name, x)
    base.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(5, activation="softmax")(x)
    return tf.keras.Model(raw, out), base


class ValMacroF1(tf.keras.callbacks.Callback):
    def __init__(self, val): super().__init__(); self.val=val; self.values=[]
    def on_epoch_end(self, epoch, logs=None):
        y=np.concatenate([y.argmax(1) for _,y in self.val]); p=self.model.predict(self.val,verbose=0).argmax(1)
        score=float(f1_score(y,p,average="macro",zero_division=0)); logs["val_macro_f1"]=score; self.values.append(score)


def weights(train):
    c=np.bincount(train.frame.label,minlength=5); return {i:len(train.frame)/(5*n) for i,n in enumerate(c)}


def write_registry(row):
    OUT.mkdir(parents=True,exist_ok=True); exists=REGISTRY.exists()
    with REGISTRY.open("a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=row.keys());
        if not exists:w.writeheader()
        w.writerow(row)


def run(spec):
    run_id="{backbone}_{augmentation}_{strategy}_{loss}_s{seed}".format(**spec); path=OUT/run_id; path.mkdir(parents=True,exist_ok=True)
    done=path/"completed.json"
    if done.exists():
        return spec | json.load(open(done))
    start=time.time(); row={"experiment_id":run_id,**spec,"status":"running","best_epoch":"","best_val_macro_f1":"","checkpoint_path":"","error_summary":""}; write_registry(row)
    try:
        set_seed(spec["seed"]); train,val,_=create_transfer_sequences(batch_size=32,seed=spec["seed"])
        model,base=build_model(spec["backbone"],spec["augmentation"])
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=.05 if spec["loss"]=="smooth" else 0.)
        cw=weights(train) if spec["loss"] in {"weighted","smooth"} else None
        metric=ValMacroF1(val); ckpt=path/"best.keras"
        callbacks=[metric,tf.keras.callbacks.ModelCheckpoint(ckpt,monitor="val_macro_f1",mode="max",save_best_only=True),tf.keras.callbacks.EarlyStopping(monitor="val_macro_f1",mode="max",patience=5,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor="val_macro_f1",mode="max",patience=3,factor=.3)]
        model.compile(tf.keras.optimizers.AdamW(1e-3,weight_decay=1e-4),loss=loss,metrics=["accuracy"])
        h=model.fit(train,validation_data=val,epochs=8 if spec["strategy"]=="F0" else 8,class_weight=cw,callbacks=callbacks,verbose=2)
        if spec["strategy"]!="F0":
            n=.25 if spec["strategy"]=="F1" else .50; start_layer=int(len(base.layers)*(1-n))
            for layer in base.layers[start_layer:]:
                if not isinstance(layer,tf.keras.layers.BatchNormalization): layer.trainable=True
            model.compile(tf.keras.optimizers.AdamW(3e-5 if spec["strategy"]=="F1" else 1e-5,weight_decay=1e-4),loss=loss,metrics=["accuracy"])
            h2=model.fit(train,validation_data=val,epochs=20,class_weight=cw,callbacks=callbacks,verbose=2); h.history={k:h.history.get(k,[])+h2.history.get(k,[]) for k in set(h.history)|set(h2.history)}
        best=max(metric.values); result={"best_val_macro_f1":best,"best_epoch":int(np.argmax(metric.values)),"seconds":time.time()-start,"params":model.count_params()}
        json.dump(spec|result,open(done,"w"),indent=2); json.dump(h.history,open(path/"history.json","w"),indent=2)
        row.update(status="completed",best_epoch=result["best_epoch"],best_val_macro_f1=best,checkpoint_path=str(ckpt));write_registry(row)
        return spec|result
    except Exception as e:
        row.update(status="failed",error_summary=repr(e));write_registry(row); traceback.print_exc()
    finally: tf.keras.backend.clear_session();gc.collect()


def select_best(results, field):
    """Choose only from completed validation results; test remains untouched."""
    valid = [result for result in results if result is not None]
    if not valid:
        raise RuntimeError("No completed result available for validation selection")
    return max(valid, key=lambda result: result["best_val_macro_f1"])[field]


def evaluate_test(spec):
    """Evaluate a frozen final configuration once and save complete test metrics."""
    run_name = "{backbone}_{augmentation}_{strategy}_{loss}_s{seed}".format(**spec)
    path = OUT / run_name
    output = path / "test_metrics.json"
    if output.exists():
        return json.load(open(output))
    _, _, test = create_transfer_sequences(batch_size=32, seed=spec["seed"])
    model = tf.keras.models.load_model(path / "best.keras", compile=False)
    y_true = np.concatenate([y.argmax(1) for _, y in test])
    y_pred = model.predict(test, verbose=0).argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(5), zero_division=0
    )
    values = {
        "experiment_id": run_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "ordinal_mae": float(np.abs(y_true-y_pred).mean()),
        "within_one_accuracy": float((np.abs(y_true-y_pred) <= 1).mean()),
        "per_class": {
            CLASSES[i]: {"precision": float(precision[i]), "recall": float(recall[i]),
                         "f1": float(f1[i]), "support": int(support[i])}
            for i in range(5)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(5)).tolist(),
    }
    json.dump(values, open(output, "w"), indent=2)
    np.savetxt(path / "test_confusion_matrix.csv", values["confusion_matrix"], delimiter=",", fmt="%d")
    tf.keras.backend.clear_session(); gc.collect()
    return values


def write_final_table(specs, test_results):
    columns = ["experiment_id", "backbone", "augmentation", "strategy", "loss", "seed", "params",
               "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
               "balanced_accuracy", "ordinal_mae", "within_one_accuracy"]
    with (OUT / "final_test_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader()
        for spec, test in zip(specs, test_results):
            completed = json.load(open(OUT / "{backbone}_{augmentation}_{strategy}_{loss}_s{seed}".format(**spec) / "completed.json"))
            writer.writerow({column: (spec | completed | test).get(column, "") for column in columns})


if __name__=="__main__":
    # Phase 1: completed checkpoints are reused, so this is safe to resume.
    screening = [
        run({"backbone": b, "augmentation": "A1", "strategy": "F1", "loss": "weighted", "seed": 42})
        for b in ["mobilenetv2", "efficientnetb0", "resnet50"]
    ]
    reference = select_best(screening, "backbone")

    # Selection below is validation-only. Test remains inaccessible until all three
    # choices are frozen.
    augmentation_results = [
        run({"backbone": reference, "augmentation": augmentation, "strategy": "F1", "loss": "weighted", "seed": 42})
        for augmentation in ["A0", "A1", "A2"]
    ]
    selected_augmentation = select_best(augmentation_results, "augmentation")

    fine_tuning_results = [
        run({"backbone": reference, "augmentation": selected_augmentation, "strategy": strategy, "loss": "weighted", "seed": 42})
        for strategy in ["F0", "F1", "F2"]
    ]
    selected_strategy = select_best(fine_tuning_results, "strategy")

    loss_results = [
        run({"backbone": reference, "augmentation": selected_augmentation, "strategy": selected_strategy, "loss": loss, "seed": 42})
        for loss in ["ce", "weighted", "smooth"]
    ]
    selected_loss = select_best(loss_results, "loss")

    final_specs = [
        {"backbone": b, "augmentation": selected_augmentation,
         "strategy": selected_strategy, "loss": selected_loss, "seed": 42}
        for b in ["mobilenetv2", "efficientnetb0", "resnet50"]
    ]
    final_results = [run(spec) for spec in final_specs]
    final_tests = [evaluate_test(spec) for spec, result in zip(final_specs, final_results) if result is not None]
    completed_final_specs = [spec for spec, result in zip(final_specs, final_results) if result is not None]
    write_final_table(completed_final_specs, final_tests)

    # Final multi-seed: seed 42 is reused, only 43/44 require additional training.
    best_backbone = select_best(final_results, "backbone")
    best_final = next(spec for spec in final_specs if spec["backbone"] == best_backbone)
    for seed in [42, 43, 44]:
        seed_spec = best_final | {"seed": seed}
        if run(seed_spec) is not None:
            evaluate_test(seed_spec)
