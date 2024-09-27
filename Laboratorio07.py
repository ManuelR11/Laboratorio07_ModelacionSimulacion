import numpy as np
import scipy.stats as stats


lambda_param = 1.0  


def valor_esperado_intervalo(a, b, num_samples=10000):
    if np.isinf(b): 

        u = np.random.uniform(0, 1, num_samples)
        x = -np.log(1 - u) / lambda_param + a 
    else:

        u = np.random.uniform(stats.expon.cdf(a, scale=1/lambda_param), stats.expon.cdf(b, scale=1/lambda_param), num_samples)
        x = -np.log(1 - u) / lambda_param

    return np.mean(x)


num_samples = 10000

# Estratos: [0, 1], [1, 3], [3, inf]
# Estrato 1: [0, 1]
p1 = stats.expon.cdf(1, scale=1/lambda_param)
esperado1 = valor_esperado_intervalo(0, 1, num_samples)

# Estrato 2: [1, 3]
p2 = stats.expon.cdf(3, scale=1/lambda_param) - stats.expon.cdf(1, scale=1/lambda_param)
esperado2 = valor_esperado_intervalo(1, 3, num_samples)

# Estrato 3: [3, inf]
p3 = 1 - stats.expon.cdf(3, scale=1/lambda_param)
esperado3 = valor_esperado_intervalo(3, np.inf, num_samples)


valor_esperado_total = p1 * esperado1 + p2 * esperado2 + p3 * esperado3

print(f"Valor esperado estimado usando estratificación: {valor_esperado_total}")
