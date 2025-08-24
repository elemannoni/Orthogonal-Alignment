import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from .merge import lerp_models, slerp_models
from .test import evaluate_model

def loss_vs_alpha_ort_al2(model1, model2, al_model2, test_dataloader):
    """"
    Confronto dell'interpolazione prima e dopo l'allineamento dei pesi 
    """
    configurations = {
        "lerp": {
            "merge_func": lerp_models, "models": (model1, model2),
            "label": "LERP Originale", "color": "blue", "linestyle": "-"
        },
        "lerp_aligned": {
            "merge_func": lerp_models, "models": (model1, al_model2),
            "label": "LERP Allineato", "color": "green", "linestyle": "-"
        },
        "slerp": {
            "merge_func": slerp_models, "models": (model1, model2),
            "label": "SLERP Originale", "color": "red", "linestyle": "--"
        },
        "slerp_aligned": {
            "merge_func": slerp_models, "models": (model1, al_model2),
            "label": "SLERP Allineato", "color": "orange", "linestyle": "--"
        },
    }

    results = {name: [] for name in configurations}
    min_values = {name: {'loss': float('inf'), 'alpha': -1} for name in configurations}
    
    alpha_values = np.linspace(0, 1, 11)

    for alpha in tqdm(alpha_values, desc="Interpolating models"):
        for name, config in configurations.items():
            m1, m2 = config["models"]
            merged_model = config["merge_func"](m1, m2, alpha)
            loss = evaluate_model(merged_model, test_dataloader)
            results[name].append(loss)
            
            if loss < min_values[name]['loss']:
                min_values[name]['loss'] = loss
                min_values[name]['alpha'] = alpha

    plt.figure(figsize=(10, 6))
    for name, config in configurations.items():
        plt.plot(
            alpha_values,
            results[name],
            marker='o',
            label=config["label"],
            color=config["color"],
            linestyle=config["linestyle"]
        )
    plt.xlabel("Alpha (α)")
    plt.ylabel("Test Loss")
    plt.title("Confronto LERP/SLERP con Allineamento del Secondo Modello")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    return (
        (min_values['lerp']['loss'], min_values['lerp']['alpha']),
        (min_values['slerp']['loss'], min_values['slerp']['alpha']),
        (min_values['lerp_aligned']['loss'], min_values['lerp_aligned']['alpha']),
        (min_values['slerp_aligned']['loss'], min_values['slerp_aligned']['alpha'])
    )

def loss_vs_alpha_ort_al1(model1, model2, al_model1, test_dataloader):
    """
    Confronto dell'interpolazione prima e dopo l'allineamento dei pesi 
    """
    configurations = {
        "lerp": {
            "merge_func": lerp_models, "models": (model1, model2),
            "label": "LERP Originale", "color": "blue", "linestyle": "-"
        },
        "lerp_aligned": {
            "merge_func": lerp_models, "models": (al_model1, model2),
            "label": "LERP Allineato", "color": "green", "linestyle": "-"
        },
        "slerp": {
            "merge_func": slerp_models, "models": (model1, model2),
            "label": "SLERP Originale", "color": "red", "linestyle": "--"
        },
        "slerp_aligned": {
            "merge_func": slerp_models, "models": (al_model1, model2),
            "label": "SLERP Allineato", "color": "orange", "linestyle": "--"
        },
    }


    results = {name: [] for name in configurations}
    min_values = {name: {'loss': float('inf'), 'alpha': -1} for name in configurations}
    
    alpha_values = np.linspace(0, 1, 11)


    for alpha in tqdm(alpha_values, desc="Interpolating models"):
        for name, config in configurations.items():
            m1, m2 = config["models"]
            merged_model = config["merge_func"](m1, m2, alpha)
            loss = evaluate_model(merged_model, test_dataloader)
            results[name].append(loss)
            
            if loss < min_values[name]['loss']:
                min_values[name]['loss'] = loss
                min_values[name]['alpha'] = alpha


    plt.figure(figsize=(10, 6))
    for name, config in configurations.items():
        plt.plot(
            alpha_values,
            results[name],
            marker='o',
            label=config["label"],
            color=config["color"],
            linestyle=config["linestyle"]
        )
    plt.xlabel("Alpha (α)")
    plt.ylabel("Test Loss")
    plt.title("Confronto LERP/SLERP con Allineamento di un Modello")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    return (
        (min_values['lerp']['loss'], min_values['lerp']['alpha']),
        (min_values['slerp']['loss'], min_values['slerp']['alpha']),
        (min_values['lerp_aligned']['loss'], min_values['lerp_aligned']['alpha']),
        (min_values['slerp_aligned']['loss'], min_values['slerp_aligned']['alpha'])
    )

def loss_vs_alpha_withperm1(model1, model2, al_model1, al_model_perm1, test_dataloader):
    """
    Confronto dell'interpolazione prima e dopo l'allineamento dei pesi (anche con permutazione)
    """
    configurations = {
        "lerp":      {"merge_func": lerp_models, "model_b": model2},
        "lerp_ort":  {"merge_func": lerp_models, "model_b": al_model1},
        "lerp_perm": {"merge_func": lerp_models, "model_b": al_model_perm1},
        "slerp":     {"merge_func": slerp_models, "model_b": model2},
        "slerp_ort": {"merge_func": slerp_models, "model_b": al_model1},
        "slerp_perm":{"merge_func": slerp_models, "model_b": al_model_perm1},
    }
    
    results = {name: [] for name in configurations}
    min_values = {
        name: {'loss': float('inf'), 'alpha': -1} for name in configurations
    }
    
    alpha_values = np.linspace(0, 1, 11)

    for alpha in tqdm(alpha_values, desc="Interpolating models"):
        for name, config in configurations.items():

            merged_model = config["merge_func"](model1, config["model_b"], alpha)

            loss = evaluate_model(merged_model, test_dataloader)
            results[name].append(loss)

            if loss < min_values[name]['loss']:
                min_values[name]['loss'] = loss
                min_values[name]['alpha'] = alpha

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(alpha_values, results["lerp"], marker='o', label="LERP Originale")
    ax1.plot(alpha_values, results["lerp_ort"], marker='o', label="LERP Ortogonale")
    ax1.plot(alpha_values, results["lerp_perm"], marker='o', label="LERP Permutazione")
    ax1.set_xlabel("Alpha (α)")
    ax1.set_ylabel("Test Loss")
    ax1.set_title("Confronto Interpolazione Lineare (LERP)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(alpha_values, results["slerp"], marker='o', label="SLERP Originale")
    ax2.plot(alpha_values, results["slerp_ort"], marker='o', label="SLERP Ortogonale")
    ax2.plot(alpha_values, results["slerp_perm"], marker='o', label="SLERP Permutazione")
    ax2.set_xlabel("Alpha (α)")
    ax2.set_ylabel("Test Loss")
    ax2.set_title("Confronto Interpolazione Sferica (SLERP)")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

    return (
        (min_values['lerp']['loss'], min_values['lerp']['alpha']),
        (min_values['slerp']['loss'], min_values['slerp']['alpha']),
        (min_values['lerp_ort']['loss'], min_values['lerp_ort']['alpha']),
        (min_values['slerp_ort']['loss'], min_values['slerp_ort']['alpha']),
        (min_values['lerp_perm']['loss'], min_values['lerp_perm']['alpha']),
        (min_values['slerp_perm']['loss'], min_values['slerp_perm']['alpha'])
    )

def loss_vs_alpha_withperm2(model1, model2, al_model2, al_model_perm2, test_dataloader):
    """
    Confronto dell'interpolazione prima e dopo l'allineamento dei pesi (anche con permutazione)
    """
    configurations = {
        "lerp": (lerp_models, model2, "C0"),
        "lerp_ort": (lerp_models, al_model2, "C1"),
        "lerp_perm": (lerp_models, al_model_perm2, "C2"),
        "slerp": (slerp_models, model2, "C0"),
        "slerp_ort": (slerp_models, al_model2, "C1"),
        "slerp_perm": (slerp_models, al_model_perm2, "C2"),
    }
    
    results = {name: [] for name in configurations}
    min_values = {
        name: {'loss': float('inf'), 'alpha': -1} for name in configurations
    }
    
    alpha_values = np.linspace(0, 1, 11)

    for alpha in tqdm(alpha_values, desc="Interpolating models"):
        for name, (merge_func, model_b, _) in configurations.items():
            merged_model = merge_func(model1, model_b, alpha)
            loss = evaluate_model(merged_model, test_dataloader)
            results[name].append(loss)

            if loss < min_values[name]['loss']:
                min_values[name]['loss'] = loss
                min_values[name]['alpha'] = alpha


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))


    ax1.plot(alpha_values, results["lerp"], marker='o', label="LERP", color=configurations["lerp"][2])
    ax1.plot(alpha_values, results["lerp_ort"], marker='o', label="LERP Ortogonale", color=configurations["lerp_ort"][2])
    ax1.plot(alpha_values, results["lerp_perm"], marker='o', label="LERP Permutazione", color=configurations["lerp_perm"][2])
    ax1.set_xlabel("Alpha (α)")
    ax1.set_ylabel("Test Loss")
    ax1.set_title("Confronto Interpolazione Lineare (LERP)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)


    ax2.plot(alpha_values, results["slerp"], marker='o', label="SLERP", color=configurations["slerp"][2])
    ax2.plot(alpha_values, results["slerp_ort"], marker='o', label="SLERP Ortogonale", color=configurations["slerp_ort"][2])
    ax2.plot(alpha_values, results["slerp_perm"], marker='o', label="SLERP Permutazione", color=configurations["slerp_perm"][2])
    ax2.set_xlabel("Alpha (α)")
    ax2.set_ylabel("Test Loss")
    ax2.set_title("Confronto Interpolazione Sferica (SLERP)")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

    return (
        (min_values['lerp']['loss'], min_values['lerp']['alpha']),
        (min_values['slerp']['loss'], min_values['slerp']['alpha']),
        (min_values['lerp_ort']['loss'], min_values['lerp_ort']['alpha']),
        (min_values['slerp_ort']['loss'], min_values['slerp_ort']['alpha']),
        (min_values['lerp_perm']['loss'], min_values['lerp_perm']['alpha']),
        (min_values['slerp_perm']['loss'], min_values['slerp_perm']['alpha'])
    )
