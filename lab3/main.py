import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd


ALPHA = 0.05

def generate_samples_variant5(seed: int = 42):
    np.random.seed(seed)

    n1, n2, n3 = 50, 200, 100

    mu1 = mu2 = mu3 = 5.0

    sigma1 = 2.0
    sigma2 = 2.0
    sigma3 = 5.0

    x1 = stats.norm.rvs(loc=mu1, scale=sigma1, size=n1)
    x2 = stats.norm.rvs(loc=mu2, scale=sigma2, size=n2)
    x3 = stats.norm.rvs(loc=mu3, scale=sigma3, size=n3)

    return x1, x2, x3

# выборочные характеристики
def sample_stats(x: np.ndarray):
    n = len(x)
    mean = np.mean(x)
    var = np.var(x, ddof=1)
    std = np.sqrt(var)
    return n, mean, var, std


def pooled_variance(samples):
    ns = np.array([len(x) for x in samples], dtype=float)
    vars_ = np.array([np.var(x, ddof=1) for x in samples], dtype=float)
    N = ns.sum()
    k = len(samples)
    sp2 = np.sum((ns - 1.0) * vars_) / (N - k)
    return sp2


def step1_descriptive_statistics(x1, x2, x3):
    print("1. ВЫБОРОЧНЫЕ ХАРАКТЕРИСТИКИ\n")

    samples = [x1, x2, x3]
    labels = ["X1", "X2", "X3"]

    stats_list = [sample_stats(x) for x in samples]
    sp2 = pooled_variance(samples)
    sp = np.sqrt(sp2)

    print("СВ\t n_i\t   x̄_i\t\t  s_i^2\t\t   s_i")
    for lbl, (n, mean, var, std) in zip(labels, stats_list):
        print(f"{lbl}\t{n:3d}\t{mean:9.4f}\t{var:9.4f}\t{std:9.4f}")
    print(f"Pooled\t---\t   ---\t{sp2:9.4f}\t{sp:9.4f}\n")

    return stats_list, sp2

# boxplot
def step2_boxplots(x1, x2, x3):
    print("2. ПОСТРОЕНИЕ BOXPLOT\n")

    plt.figure(figsize=(6, 4))
    plt.boxplot([x1, x2, x3], labels=["X1", "X2", "X3"])
    plt.title("Диаграммы Box-and-Whisker для X1, X2, X3")
    plt.xlabel("Группа")
    plt.ylabel("Значения")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

# критерий Бартлетта
def step3_bartlett_test(x1, x2, x3, alpha=ALPHA):
    print("3. ПРОВЕРКА УСЛОВИЯ ПРИМЕНИМОСТИ\n")
    stat, p_value = stats.bartlett(x1, x2, x3)

    print(f"Статистика Бартлетта: T_B = {stat:.4f}")
    print(f"p-value = {p_value:.6g}")

    if p_value < alpha:
        decision = (
            f"При уровне значимости α = {alpha} гипотеза H0 "
            f"о равенстве дисперсий отвергается."
        )
    else:
        decision = (
            f"При уровне значимости α = {alpha} нет оснований отвергнуть гипотезу H0 "
            f"о равенстве дисперсий."
        )

    print(decision + "\n")
    return stat, p_value

# однофакторный дисперсионный анализ
def step4_one_way_anova(x1, x2, x3, alpha=ALPHA):
    print("4. ОДНОФАКТОРНЫЙ ДИСПЕРСИОННЫЙ АНАЛИЗ\n")

    samples = [x1, x2, x3]
    data = np.concatenate(samples)

    means = np.array([np.mean(x) for x in samples])
    ns = np.array([len(x) for x in samples], dtype=float)
    overall_mean = np.mean(data)

    SS_between = np.sum(ns * (means - overall_mean) ** 2)
    SS_within = sum(((x - m) ** 2).sum() for x, m in zip(samples, means))
    SS_total = ((data - overall_mean) ** 2).sum()

    k = len(samples)
    N = len(data)

    df_between = k - 1
    df_within = N - k
    df_total = N - 1

    MS_between = SS_between / df_between
    MS_within = SS_within / df_within

    F_emp = MS_between / MS_within
    p_value = stats.f.sf(F_emp, df_between, df_within)

    R2 = SS_between / SS_total
    eta = np.sqrt(R2)

    print("Источник вариации\tПоказатель вариации\t\t Число степеней свободы\t Несмещенная оценка\t\t")
    print(
        f"Группировочный признак\t{SS_between:9.4f}\t{df_between:3d}\t"
        f"{MS_between:9.4f}\t"
    )
    print(
        f"Остаточные признаки\t{SS_within:9.4f}\t{df_within:3d}\t"
        f"{MS_within:9.4f}\t"
    )
    print(
        f"Все признаки\t\t{SS_total:9.4f}\t{df_total:3d}\t"
        f"    ---\t\t\n"
    )

    print(f"Эмпирический коэффициент детерминации R^2 = {R2:.4f}")
    print(f"Эмпирическое корреляционное отношение η = {eta:.4f}\n")

    print(f"Статистика критерия Фишера: F = {F_emp:.4f}")
    print(f"p-value = {p_value:.6g}")

    if p_value < alpha:
        decision = (
            f"При уровне значимости α = {alpha} гипотеза H0 "
            f"о равенстве математических ожиданий ОТВЕРГАЕТСЯ."
        )
    else:
        decision = (
            f"При уровне значимости α = {alpha} нет оснований отвергнуть гипотезу H0 "
            f"о равенстве математических ожиданий."
        )

    print(decision + "\n")

    return {
        "SS_between": SS_between,
        "SS_within": SS_within,
        "SS_total": SS_total,
        "df_between": df_between,
        "df_within": df_within,
        "df_total": df_total,
        "MS_between": MS_between,
        "MS_within": MS_within,
        "F": F_emp,
        "p_value": p_value,
        "R2": R2,
        "eta": eta,
        "means": means,
        "ns": ns,
    }

# метод линейных контрастов
def step5_linear_contrasts(anova_res, alpha=ALPHA):
    means = anova_res["means"]
    ns = anova_res["ns"]
    df_within = anova_res["df_within"]
    MS_within = anova_res["MS_within"]

    sp = np.sqrt(MS_within)
    t_crit = stats.t.ppf(1 - alpha / 2, df_within)

    labels = ["X1", "X2", "X3"]

    print("5. МЕТОД ЛИНЕЙНЫХ КОНТРАСТОВ\n")


def tukey_multiple_comparisons(x1, x2, x3, alpha=ALPHA):
    labels = ["X1", "X2", "X3"]
    data = np.concatenate([x1, x2, x3])
    group_labels = sum(([lbl] * len(x) for lbl, x in zip(labels, [x1, x2, x3])), [])

    tukey = pairwise_tukeyhsd(endog=data, groups=group_labels, alpha=alpha)
    print("ПОПАРНЫЕ СРАВНЕНИЯ:\n")
    print(tukey)
    print()
    tukey.plot_simultaneous()
    plt.title("Tukey HSD: доверительные интервалы для разностей средних")
    plt.show()


def main():
    x1, x2, x3 = generate_samples_variant5(seed=42)  # seed можно поменять

    step1_descriptive_statistics(x1, x2, x3)

    step2_boxplots(x1, x2, x3)

    step3_bartlett_test(x1, x2, x3, alpha=ALPHA)

    anova_res = step4_one_way_anova(x1, x2, x3, alpha=ALPHA)

    step5_linear_contrasts(anova_res, alpha=ALPHA)
    tukey_multiple_comparisons(x1, x2, x3, alpha=ALPHA)


if __name__ == "__main__":
    main()
