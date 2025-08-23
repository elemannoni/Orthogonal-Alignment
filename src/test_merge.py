import matplotlib.pyplot as plt
from tqdm import tqdm

def loss_vs_alpha_ort_al2(model1, model2, al_model2, test_dataloader):
    lista_lerp = []
    lista_slerp = []
    lista_lerp_aligned = []
    lista_slerp_aligned = []
    min = 0
    min_alpha = 0
    min_s = 0
    mins_alpha = 0
    min_al = 0
    min_al_alpha = 0
    min_al_s = 0
    min_als_alpha = 0
    alpha_values = [i/10 for i in range(11)]
    for alpha in tqdm(alpha_values):
      merge = evaluate_model(lerp_models(model1, model2, alpha), test_dataloader)
      if merge < min:
        min = merge
        min_alpha = alpha
      lista_lerp.append(merge)
      merge_s = evaluate_model(slerp_models(model1, model2, alpha), test_dataloader)
      if merge_s < min_s:
        min_s = merge_s
        mins_alpha = alpha
      lista_slerp.append(merge_s)
      merge_al = evaluate_model(lerp_models(model1, al_model2, alpha), test_dataloader)
      if merge_al < min_al:
        min_al = merge_al
        min_al_alpha = alpha
      lista_lerp_aligned.append(merge_al)
      merge_al_s = evaluate_model(slerp_models(model1, al_model2, alpha), test_dataloader)
      if merge_al_s < min_al_s:
        min_al_s = merge_al_s
        min_als_alpha = alpha
      lista_slerp_aligned.append(merge_al_s)
    plt.figure(figsize=(10,5))

    plt.plot(alpha_values,lista_lerp, marker='o')
    plt.plot(alpha_values,lista_lerp_aligned, marker='o')
    plt.plot(alpha_values,lista_slerp, marker='o')
    plt.plot(alpha_values,lista_slerp_aligned, marker='o')
    plt.xlabel("alpha")
    plt.ylabel("loss")
    plt.legend(["lerp", "lerp_al", "slerp", "slerp_al"])
    plt.show()

    return (min, min_alpha), (min_s, mins_alpha), (min_al, min_al_alpha), (min_al_s, min_als_alpha)


def loss_vs_alpha_ort_al1(model1, model2, al_model1, test_dataloader):
    lista_lerp = []
    lista_slerp = []
    lista_lerp_aligned = []
    lista_slerp_aligned = []
    min = 0
    min_alpha = 0
    min_s = 0
    mins_alpha = 0
    min_al = 0
    min_al_alpha = 0
    min_al_s = 0
    min_als_alpha = 0
    alpha_values = [i/10 for i in range(11)]
    for alpha in tqdm(alpha_values):
      merge = evaluate_model(lerp_models(model1, model2, alpha), test_dataloader)
      if merge < min:
        min = merge
        min_alpha = alpha
      lista_lerp.append(merge)
      merge_s = evaluate_model(slerp_models(model1, model2, alpha), test_dataloader)
      if merge_s < min_s:
        min_s = merge_s
        mins_alpha = alpha
      lista_slerp.append(merge_s)
      merge_al = evaluate_model(lerp_models(al_model1, model2, alpha), test_dataloader)
      if merge_al < min_al:
        min_al = merge_al
        min_al_alpha = alpha
      lista_lerp_aligned.append(merge_al)
      merge_al_s = evaluate_model(slerp_models(al_model1, model2, alpha), test_dataloader)
      if merge_al_s < min_al_s:
        min_al_s = merge_al_s
        min_als_alpha = alpha
      lista_slerp_aligned.append(merge_al_s)

    plt.figure(figsize=(10,5))

    plt.plot(alpha_values,lista_lerp, marker='o')
    plt.plot(alpha_values,lista_lerp_aligned, marker='o')
    plt.plot(alpha_values,lista_slerp, marker='o')
    plt.plot(alpha_values,lista_slerp_aligned, marker='o')
    plt.xlabel("alpha")
    plt.ylabel("loss")
    plt.legend(["lerp", "lerp_al", "slerp", "slerp_al"])
    plt.show()

    return (min, min_alpha), (min_s, mins_alpha), (min_al, min_al_alpha), (min_al_s, min_als_alpha)
  

def loss_vs_alpha_withperm1(model1, model2, al_model1, al_model_perm1, test_dataloader):
    lista_lerp = []
    lista_slerp = []
    lista_lerp_aligned = []
    lista_slerp_aligned = []
    lista_lerp_aligned_perm = []
    lista_slerp_aligned_perm = []
    min = 0
    min_alpha = 0
    min_s = 0
    mins_alpha = 0
    min_al = 0
    min_al_alpha = 0
    min_al_s = 0
    min_als_alpha = 0
    min_perm = 0
    min_perm_alpha = 0
    min_perm_s = 0
    min_perm_s_alpha = 0
    alpha_values = [i/10 for i in range(11)]
    for alpha in tqdm(alpha_values):
      merge = evaluate_model(lerp_models(model1, model2, alpha), test_dataloader)
      if merge < min:
        min = merge
        min_alpha = alpha
      lista_lerp.append(merge)
      merge_s = evaluate_model(slerp_models(model1, model2, alpha), test_dataloader)
      if merge_s < min_s:
        min_s = merge_s
        mins_alpha = alpha
      lista_slerp.append(merge_s)
      merge_al = evaluate_model(lerp_models(al_model1, model2, alpha), test_dataloader)
      if merge_al < min_al:
        min_al = merge_al
        min_al_alpha = alpha
      lista_lerp_aligned.append(merge_al)
      merge_al_s = evaluate_model(slerp_models(al_model1, model2, alpha), test_dataloader)
      if merge_al_s < min_al_s:
        min_al_s = merge_al_s
        min_als_alpha = alpha
      lista_slerp_aligned.append(merge_al_s)
      merge_perm = evaluate_model(lerp_models(al_model_perm1, model2, alpha), test_dataloader)
      if merge_perm < min_perm:
        min_perm = merge_perm
        min_perm_alpha = alpha
      lista_lerp_aligned_perm.append(merge_perm)
      merge_perm_s = evaluate_model(slerp_models(al_model_perm1, model2, alpha), test_dataloader)
      if merge_perm_s < min_perm_s:
        min_perm_s = merge_perm_s
        min_perm_s_alpha = alpha
      lista_slerp_aligned_perm.append(merge_perm_s)

    plt.figure(figsize=(10,5))

    #LERP
    plt.subplot(1, 2, 1)
    plt.plot(alpha_values,lista_lerp, marker='o')
    plt.plot(alpha_values,lista_lerp_aligned, marker='o')
    plt.plot(alpha_values,lista_lerp_aligned_perm, marker='o')
    plt.xlabel("alpha")
    plt.ylabel("loss")
    plt.legend(["lerp", "lerp_ort", "lerp_perm"])
    plt.title("LERP")
    plt.show()

    #SLERP
    plt.subplot(1, 2, 2)
    plt.plot(alpha_values,lista_slerp, marker='o')
    plt.plot(alpha_values,lista_slerp_aligned, marker='o')
    plt.plot(alpha_values,lista_slerp_aligned_perm, marker='o')
    plt.xlabel("alpha")
    plt.ylabel("loss")
    plt.legend(["slerp", "slerp_ort", "slerp_perm"])
    plt.title("SLERP")
    plt.show()

    return (min, min_alpha), (min_s, mins_alpha), (min_al, min_al_alpha), (min_al_s, min_als_alpha), (min_perm, min_perm_alpha), (min_perm_s, min_perm_s_alpha)


def loss_vs_alpha_withperm2(model1, model2, al_model2, al_model_perm2, test_dataloader):
    lista_lerp = []
    lista_slerp = []
    lista_lerp_aligned = []
    lista_slerp_aligned = []
    lista_lerp_aligned_perm = []
    lista_slerp_aligned_perm = []
    min = 0
    min_alpha = 0
    min_s = 0
    mins_alpha = 0
    min_al = 0
    min_al_alpha = 0
    min_al_s = 0
    min_als_alpha = 0
    min_perm = 0
    min_perm_alpha = 0
    min_perm_s = 0
    min_perm_s_alpha = 0
    alpha_values = [i/10 for i in range(11)]
    for alpha in tqdm(alpha_values):
      merge = evaluate_model(lerp_models(model1, model2, alpha), test_dataloader)
      if merge < min:
        min = merge
        min_alpha = alpha
      lista_lerp.append(merge)
      merge_s = evaluate_model(slerp_models(model1, model2, alpha), test_dataloader)
      if merge_s < min_s:
        min_s = merge_s
        mins_alpha = alpha
      lista_slerp.append(merge_s)
      merge_al = evaluate_model(lerp_models(model1, al_model2, alpha), test_dataloader)
      if merge_al < min_al:
        min_al = merge_al
        min_al_alpha = alpha
      lista_lerp_aligned.append(merge_al)
      merge_al_s = evaluate_model(slerp_models(model1, al_model2, alpha), test_dataloader)
      if merge_al_s < min_al_s:
        min_al_s = merge_al_s
        min_als_alpha = alpha
      lista_slerp_aligned.append(merge_al_s)
      merge_perm = evaluate_model(lerp_models(model1, al_model_perm2, alpha), test_dataloader)
      if merge_perm < min_perm:
        min_perm = merge_perm
        min_perm_alpha = alpha
      lista_lerp_aligned_perm.append(merge_perm)
      merge_perm_s = evaluate_model(slerp_models(model1, al_model_perm2, alpha), test_dataloader)
      if merge_perm_s < min_perm_s:
        min_perm_s = merge_perm_s
        min_perm_s_alpha = alpha
      lista_slerp_aligned_perm.append(merge_perm_s)

    plt.figure(figsize=(10,5))

    #LERP
    plt.subplot(1, 2, 1)
    plt.plot(alpha_values,lista_lerp, marker='o')
    plt.plot(alpha_values,lista_lerp_aligned, marker='o')
    plt.plot(alpha_values,lista_lerp_aligned_perm, marker='o')
    plt.xlabel("alpha")
    plt.ylabel("loss")
    plt.legend(["lerp", "lerp_ort", "lerp_perm"])
    plt.title("LERP")
    plt.show()

    #SLERP
    plt.subplot(1, 2, 2)
    plt.plot(alpha_values,lista_slerp, marker='o')
    plt.plot(alpha_values,lista_slerp_aligned, marker='o')
    plt.plot(alpha_values,lista_slerp_aligned_perm, marker='o')
    plt.xlabel("alpha")
    plt.ylabel("loss")
    plt.legend(["slerp", "slerp_ort", "slerp_perm"])
    plt.title("SLERP")
    plt.show()

    return (min, min_alpha), (min_s, mins_alpha), (min_al, min_al_alpha), (min_al_s, min_als_alpha), (min_perm, min_perm_alpha), (min_perm_s, min_perm_s_alpha)

