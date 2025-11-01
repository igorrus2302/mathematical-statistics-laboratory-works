import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# дано
np.random.seed(5)
n1 = 100
n2 = 100
mu = 5
sigma = 2

# выборки для варианта 5
X = np.random.normal(loc=mu, scale=sigma, size=n1)
Y = np.random.normal(loc=mu, scale=sigma, size=n2)

# выборочные характеристики
x_mean = X.mean()
x_var = X.var(ddof=1)
x_std = X.std(ddof=1)

y_mean = Y.mean()
y_var = Y.var(ddof=1)
y_std = Y.std(ddof=1)

print("=== 1. Выборочные характеристики ===")
print(f"X: mean = {x_mean:.4f}, s^2 = {x_var:.4f}, s = {x_std:.4f}")
print(f"Y: mean = {y_mean:.4f}, s^2 = {y_var:.4f}, s = {y_std:.4f}")

# вспомогательная функция ECDF
def ecdf(sample):
    sample = np.sort(sample)
    n = len(sample)
    y = np.arange(1, n + 1) / n
    return sample, y

# гистограммы X
print("\n=== 2. Гистограммы X ===")
bins_list = [5, 10, 15, 20]
for bins in bins_list:
    plt.figure()
    plt.hist(X, bins=bins, edgecolor="black")
    plt.title(f"Гистограмма X, NBins = {bins}")
    plt.xlabel("x")
    plt.ylabel("частота")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"hist_X_{bins}.png")
    plt.close()
    print(f"hist_X_{bins}.png сохранена")

# критерий хи-квадрат (3 гипотезы)
def chi_square_gof(sample, cdf, bins, params_known=0, a=None, b=None):
    n = len(sample)
    if a is None:
        a = sample.min()
    if b is None:
        b = sample.max()

    edges = np.linspace(a, b, bins + 1)
    obs, _ = np.histogram(sample, bins=edges)
    F = cdf(edges)
    probs = np.diff(F)
    exp = n * probs
    exp = np.where(exp == 0, 1e-12, exp)

    chi2 = ((obs - exp) ** 2 / exp).sum()
    df = bins - 1 - params_known
    pval = 1 - stats.chi2.cdf(chi2, df)
    return chi2, pval, df

# гипотеза: нормальное
def normal_cdf(x):
    return stats.norm.cdf(x, loc=mu, scale=sigma)

# гипотеза: равномерное с оценёнными границами
a_hat = X.min()
b_hat = X.max()
def uniform_cdf(x):
    return stats.uniform.cdf(x, loc=a_hat, scale=(b_hat - a_hat))

# гипотеза: хи-квадрат с 5 степенями свободы
def chi2_5_cdf(x):
    return stats.chi2.cdf(x, df=5)

print("\n=== 3. Критерий хи-квадрат ===")
for bins in [5, 10, 15, 20]:
    chi2_val, p_val, df = chi_square_gof(X, normal_cdf, bins, params_known=0)
    print(f"H0: X ~ N(5,2), k={bins:2d}: chi2 = {chi2_val:.4f}, p = {p_val:.4f}")

for bins in [5, 10, 15, 20]:
    chi2_val, p_val, df = chi_square_gof(X, uniform_cdf, bins, params_known=2)
    print(f"H0: X ~ U[a_hat,b_hat], k={bins:2d}: chi2 = {chi2_val:.4f}, p = {p_val:.3e}")

for bins in [5, 10, 15, 20]:
    chi2_val, p_val, df = chi_square_gof(
        X, chi2_5_cdf, bins, params_known=0,
        a=0.0, b=X.max()
    )
    print(f"H0: X ~ chi2(5), k={bins:2d}: chi2 = {chi2_val:.4f}, p = {p_val:.3e}")

# критерий Колмогорова
print("\n=== 4. Критерий Колмогорова ===")
ks_norm = stats.kstest(X, "norm", args=(mu, sigma))
ks_unif = stats.kstest(X, "uniform", args=(a_hat, b_hat - a_hat))
ks_chi2 = stats.kstest(X, "chi2", args=(5,))

print(f"H0: X ~ N(5,2): D = {ks_norm.statistic:.4f}, p = {ks_norm.pvalue:.4f}")
print(f"H0: X ~ U[a_hat,b_hat]: D = {ks_unif.statistic:.4f}, p = {ks_unif.pvalue:.3e}")
print(f"H0: X ~ chi2(5): D = {ks_chi2.statistic:.4f}, p = {ks_chi2.pvalue:.3e}")

# график ECDF + теоретические
x_sorted, Fx = ecdf(X)
x_grid = np.linspace(min(0.0, X.min()), X.max(), 400)

plt.figure()
plt.step(x_sorted, Fx, where="post", label="ECDF X")
plt.plot(x_grid, stats.norm.cdf(x_grid, mu, sigma), label="N(5,2)")
plt.plot(x_grid, stats.uniform.cdf(x_grid, a_hat, b_hat - a_hat), label="U[a_hat,b_hat]")
plt.plot(x_grid, stats.chi2.cdf(x_grid, 5), label="chi2(5)")
plt.xlabel("x")
plt.ylabel("F(x)")
plt.title("Эмпирическая и теоретические ФР")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("cdf_X_all.png")
plt.close()
print("cdf_X_all.png сохранена")

# 5. двухвыборочные критерии
print("\n=== 5. Двухвыборочные критерии (X vs Y) ===")

def chi2_homogeneity(x, y, bins=5):
    data = np.concatenate([x, y])
    edges = np.linspace(data.min(), data.max(), bins + 1)
    x_counts, _ = np.histogram(x, bins=edges)
    y_counts, _ = np.histogram(y, bins=edges)
    table = np.vstack([x_counts, y_counts])
    chi2, p, dof, expected = stats.chi2_contingency(table)
    return chi2, p, dof

chi2_xy, p_xy, df_xy = chi2_homogeneity(X, Y, bins=5)
print(f"Chi-squared (однородность, 5 интервалов): chi2 = {chi2_xy:.4f}, p = {p_xy:.4f}")

ks2 = stats.ks_2samp(X, Y, method="auto")
print(f"KS двухвыборочный: D = {ks2.statistic:.4f}, p = {ks2.pvalue:.4f}")

diff = X - Y
diff_nz = diff[diff != 0]
n_pos = np.sum(diff_nz > 0)
n_neg = np.sum(diff_nz < 0)
sign_res = stats.binomtest(n_pos, n_pos + n_neg, 0.5, alternative="two-sided")
print(f"Sign test: n+ = {n_pos}, n- = {n_neg}, p = {sign_res.pvalue:.4f}")

u_res = stats.mannwhitneyu(X, Y, alternative="two-sided")
print(f"Mann–Whitney U: U = {u_res.statistic:.1f}, p = {u_res.pvalue:.4f}")

# гистограммы X и Y
plt.figure()
plt.hist(X, bins=15, alpha=0.5, edgecolor="black", label="X")
plt.hist(Y, bins=15, alpha=0.5, edgecolor="black", label="Y")
plt.title("Гистограммы X и Y")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("hist_XY.png")
plt.close()
print("hist_XY.png сохранена")

# ECDF X и Y
x_sorted, Fx = ecdf(X)
y_sorted, Fy = ecdf(Y)
plt.figure()
plt.step(x_sorted, Fx, where="post", label="ECDF X")
plt.step(y_sorted, Fy, where="post", label="ECDF Y")
plt.title("ECDF: X и Y")
plt.xlabel("x")
plt.ylabel("F(x)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("ecdf_XY.png")
plt.close()
print("ecdf_XY.png сохранена")

