import yaml
import itertools
import subprocess
import shutil

BASE_CONFIG = "config/dqn.yaml"
BACKUP_CONFIG = "config/dqn_backup.yaml"

grid = {
    "learning_rate": [2e-5],
    "epsilon_decay": [0.9993, 0.9995],
    #"epsilon_end": [0.05],
    "beta_qos": [5.0, 10.0],
    "n_levels": [8, 10],
}


# Sauvegarde du vrai dqn.yaml
shutil.copy(BASE_CONFIG, BACKUP_CONFIG)

keys = list(grid.keys())
configs = list(itertools.product(*grid.values()))

for i, combo in enumerate(configs):
    print(f"\n🔁 Run {i+1}/{len(configs)}")
    print(dict(zip(keys, combo)))

    with open(BACKUP_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)

    for k, v in zip(keys, combo):
        if k == "learning_rate":
            cfg["dqn"]["training"]["learning_rate"] = v
        elif k == "epsilon_decay":
            cfg["dqn"]["training"]["epsilon_decay"] = v
        elif k == "beta_qos":
            cfg["dqn"]["reward"]["beta_qos"] = v
        elif k == "n_levels":
            cfg["dqn"]["actions"]["n_levels"] = v

    # Écrase dqn.yaml
    with open(BASE_CONFIG, "w") as f:
        yaml.dump(cfg, f)

    # Lancer entraînement court
    subprocess.run([
        "python", "scripts/train.py",
        "--episodes", "800"
    ])

# Restaurer config originale
shutil.copy(BACKUP_CONFIG, BASE_CONFIG)
print("\n✅ Grid search terminée, config restaurée.")
