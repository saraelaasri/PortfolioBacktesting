import re
import pandas as pd
import yfinance as yf
from functools import reduce
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import matplotlib
import pandas as pd
import math
import numpy as np
import seaborn as sb
import df2img
from scipy . stats import norm
from matplotlib import pyplot as plt
import yfinance as yf
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from scipy import optimize
from scipy.stats import shapiro
from datetime import datetime
import seaborn as sns
import plotly.graph_objects as go
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly_express as px
import pandas_datareader as web
from datetime import datetime as dt, timedelta as td
from pypfopt import expected_returns, risk_models

import os
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64
from functools import reduce
import time
import requests
from tabulate import tabulate
from PIL import Image
from datetime import timedelta
import matplotlib.backends.backend_tkagg
matplotlib.use('tkagg')
import quandl
import scipy.optimize as sco
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy as np

date_de_debut = '2022-04-1'
date_de_fin = '2023-04-01'
tickers = ['AAPL', 'AMZN', 'PYPL', 'BA', 'ECL']
anticipated_returns = np.array([
    0.0044317979673250425,  # AAPL
    0.009838164734066948,   # AMZN
    -0.0016615889846242327,  # PYPL
    -0.0019060731864268424,  # BA
    -0.001161729013802585   # ECL
])  # user views

def fetch_adj_close(symbol):
    API_KEY = 'ZS0Y1JTMURGXWJUM'
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={API_KEY}&outputsize=full'
    response = requests.get(url)
    data = response.json()
    if 'Time Series (Daily)' in data:
        daily_data = data['Time Series (Daily)']
        df = pd.DataFrame(daily_data).T
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[['5. adjusted close']]
        df.columns = [symbol]
        return df
    else:
        print("Error fetching data:", data)
        return None
#premiere function
#premiere function
def get_filtered_prices(tickers, date_de_debut, date_de_fin):
    adj_prices = []
    for i in range(len(tickers)):
        close_price = pd.DataFrame(yf.download(tickers[i])['Adj Close'].dropna(axis=0, how='any'))
        time.sleep(4)
        close_price = close_price.loc[~close_price.index.duplicated(keep='last')]
        close_price.columns = [tickers[i]]
        adj_prices.append(close_price)

    df = reduce(lambda x, y: pd.merge(x, y, left_index=True,
                                      right_index=True, how='outer'), adj_prices)
    df.sort_index(ascending=False, inplace=True)
    df.index = pd.to_datetime(df.index).date
    df = df.rename_axis('Date').reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index(['Date'])

    filtered_df = df.loc[date_de_debut:date_de_fin]
    filtered_df = filtered_df.dropna()

    return filtered_df

def create_filtered_prices_graph(dataa):
    fig, ax = plt.subplots(figsize=(15, 7))  # Adjust the figsize to desired dimensions
    dataa.plot(ax=ax)

    plt.title("Comparaison du prix des indices", fontsize=16)
    plt.ylabel('Prix', fontsize=14)
    plt.xlabel('Année', fontsize=14)

    plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)

    # Save the figure to an in-memory buffer
    image_buffer = io.BytesIO()
    plt.savefig(image_buffer, format='png')

    image_buffer.seek(0)

    # Convert the image buffer to a base64-encoded string
    image_base64 = base64.b64encode(image_buffer.getvalue()).decode()

    return image_base64


def dtrgraph(dataa):
    filtered_df = dataa

    dtr = filtered_df.pct_change()
    data = dtr.dropna()

    fig, ax = plt.subplots(figsize=(6, 6))  # Adjust the figsize to desired dimensions
    data.plot(ax=ax)

    plt.title("Comparaison du prix des rendements", fontsize=16)
    plt.ylabel('Prix', fontsize=14)
    plt.xlabel('Année', fontsize=14)

    plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)

    image_buffer = io.BytesIO()
    plt.savefig(image_buffer, format='png')
    image_buffer.seek(0)
    image_base64 = base64.b64encode(image_buffer.getvalue()).decode()

    return image_base64



def plot_correlation_heatmap(dataa):
    filtered_df = dataa

    dtr = filtered_df.pct_change()
    data = dtr.dropna()
    corr_df = data.corr(method='pearson')

    fig, ax = plt.subplots(figsize=(15, 7))  # Adjust the figsize to desired dimensions
    sns.heatmap(corr_df, annot=True, ax=ax)

    image_buffer = io.BytesIO()
    plt.savefig(image_buffer, format='png')
    plt.close()

    image_buffer.seek(0)
    image_base64 = base64.b64encode(image_buffer.getvalue()).decode()

    return image_base64
# dtr = dtrgraph(lambda: get_filtered_prices(tickers, date_de_debut, date_de_fin))


def describe_prices(dataa):

    dtr = dataa.pct_change()
    dtr = dtr.dropna()

    description = dtr.describe()

    # Convert the description table to a text-based table

    return description

#CORRELATION



def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) *252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns
def random_portfolios(num_portfolios, mean_returns, cov_matrix):
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(5)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return -0.0387) / portfolio_std_dev
    return results, weights_record
def neg_sharpe_ratio(weights, mean_returns, cov_matrix):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret -0.0387)/ p_var
def max_sharpe_ratio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,0.6)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]

def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,0.6)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    return result

def efficient_return(mean_returns, cov_matrix, target):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients


def MV(data,mean_returns, cov_matrix, num_portfolios):
    results, _ = random_portfolios(num_portfolios, mean_returns, cov_matrix)

    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix)
    sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x, index=data.columns, columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i * 100, 2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    return max_sharpe_allocation.T / 100


def display_calculated_ef_with_random(dataa, num_portfolios):
    filtered_df = dataa  # Call the get_filtered_prices_func function

    dtr = filtered_df.pct_change()
    dtr = dtr.dropna()

    mean_returns = dtr.mean()
    cov_matrix = dtr.cov()

    results, _ = random_portfolios(num_portfolios, mean_returns, cov_matrix)

    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix)
    sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x, index=dtr.columns, columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i * 100, 2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(min_vol.x, index=dtr.columns, columns=['allocation'])
    min_vol_allocation.allocation = [round(i * 100, 2) for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T

    fig, ax = plt.subplots(figsize=(15, 8))
    scatter = ax.scatter(results[0, :], results[1, :], c=results[2, :], cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar(scatter)  # Use the scatter object as the mappable for the colorbar
    ax.scatter(sdp, rp, marker='*', color='r', s=500, label='Ratio de Sharpe maximum')
    ax.scatter(sdp_min, rp_min, marker='*', color='g', s=500, label='Volatilité minimale')

    target = np.linspace(rp_min, 0.32, 50)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
    ax.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='Frontière efficace')
    ax.set_title("Optimisation du portefeuille basée sur la frontière efficace")
    ax.set_xlabel('Volatilité annuelle')
    ax.set_ylabel('Rendements annuels')
    ax.legend(labelspacing=0.8)

    # Save the plot to an in-memory buffer
    image_buffer = io.BytesIO()
    plt.savefig(image_buffer, format='png')
    plt.close()

    # Convert the image buffer to a base64-encoded string
    image_buffer.seek(0)
    image_base64 = base64.b64encode(image_buffer.getvalue()).decode()

    return image_base64


import numpy as np
from numpy.linalg import inv

def black_litterman_model(cov, rendement, anticipated_returns, tau_range=np.linspace(0.001, 1, 100)):
    # Rentabilités d'équilibre
    rendements_equilibre_implicites = rendement.mean() * (1 / 5)
    # Initialiser le vecteur q en soustrayant les rentabilités d'équilibre des rentabilités anticipées
    q = anticipated_returns - rendements_equilibre_implicites
    # Définir les variables
    P = np.eye(len(anticipated_returns))  # adjusted to match the length of anticipated_returns

    def calculate_difference(tau, cov, rendements_equilibre_implicites, P, Omega, q):
        matrice_cov_ajustee = tau * cov
        vecteur_rendement_BL = rendements_equilibre_implicites + matrice_cov_ajustee.dot(P.T).dot(
            inv(P.dot(matrice_cov_ajustee).dot(P.T) + Omega).dot(q - P.dot(rendements_equilibre_implicites)))
        return np.sum(np.abs(vecteur_rendement_BL - rendements_equilibre_implicites))

    best_tau = None
    min_difference = None
    for tau in tau_range:
        Omega = np.diag(np.matmul(P, np.matmul(tau * cov, P.T)))
        difference = calculate_difference(tau, cov, rendements_equilibre_implicites, P, Omega, q)

        if min_difference is None or difference < min_difference:
            best_tau = tau
            min_difference = difference

    # With the best tau found, calculate the Black-Litterman return vector and weights
    tau = best_tau
    matrice_cov_ajustee = tau * cov
    Omega = np.diag(np.matmul(P, np.matmul(matrice_cov_ajustee, P.T)))
    vecteur_rendement_BL = rendements_equilibre_implicites + matrice_cov_ajustee.dot(P.T).dot(
        inv(P.dot(matrice_cov_ajustee).dot(P.T) + Omega).dot(q - P.dot(rendements_equilibre_implicites)))
    weights_BL = np.matmul(inv(tau * cov), vecteur_rendement_BL)
    weights_BL_normalized = weights_BL / np.sum(weights_BL)
    sum_weights = weights_BL_normalized.sum()

    # Vérifier que chaque poids est compris entre 0% et 100%
    min_weight = weights_BL_normalized.min()
    max_weight = weights_BL_normalized.max()

    # Contraindre les poids pour qu'ils soient compris entre 0% et 100%
    weights_BL_normalized = np.clip(weights_BL_normalized, 0, 60)

    # Rééquilibrer les poids pour qu'ils somment à 1 (ou 100%)
    weights_BL_df = weights_BL_normalized / weights_BL_normalized.sum()
    return weights_BL_df


def fitness_function(weights, data_returns, risk_free_rate):
    weights = np.array(weights)
    portfolio_returns = np.sum(data_returns * weights, axis=1)
    annualized_portfolio_returns = portfolio_returns * 252
    portfolio_mean = np.mean(annualized_portfolio_returns)
    portfolio_std = np.std(annualized_portfolio_returns) * np.sqrt(252)
    portfolio_beta = np.cov(portfolio_returns, data_returns[:, 0])[0, 1] / np.var(data_returns[:, 0])

    # Calmar Ratio
    portfolio_cum_returns = np.cumprod(1 + annualized_portfolio_returns) - 1
    max_drawdown = np.nanmax(np.maximum.accumulate(portfolio_cum_returns) - portfolio_cum_returns)
    calmar_ratio = portfolio_mean / (max_drawdown + 1e-8)

    sharpe_ratio = (portfolio_mean - risk_free_rate) / (portfolio_std + 1e-8)
    treynor_ratio = (portfolio_mean - risk_free_rate) / (portfolio_beta + 1e-8)

    # Multi-factor fitness score
    fitness_score = sharpe_ratio + calmar_ratio + treynor_ratio
    weight_std = np.std(weights)
    diversification_penalty = 1 - weight_std
    fitness_score_with_penalty = fitness_score * diversification_penalty

    return fitness_score_with_penalty
def genetic_algorithm(data, population_size=500, num_generations=100, mutation_rate=0.1, elitism=0.2,risk_free_rate=0.01):
    #Créer une population aléatoire de vecteurs de poids de portefeuille
    #représentant des solutions candidates au problème d'optimisation de portefeuille.
    population = np.random.rand(population_size, len(data.columns))
    #Cette ligne de code normalise les poids de chaque individu dans
    #la population de sorte que la somme des poids de chaque individu soit égale à 1.
    population = population / np.sum(population, axis=1)[:, np.newaxis]

    # la valeur de ratio sharp pour chaque individu de la population
    fitness = np.array([fitness_function(individual,data.to_numpy(), risk_free_rate) for individual in population])

    for generation in range(num_generations):
        #trier la population en fonction de la valeur de ratio sharp
        sorted_idx = np.argsort(fitness)[::-1]
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]
        # la proportion des meilleurs individus de la population actuelle à conserver sans modification
        #pour la génération suivante.
        num_elites = int(elitism * population_size)
        #Crée la génération suivante en commençant par inclure les individus élites de la population actuelle.
        offspring = population[:num_elites]
        #Les indices des parents sont choisis au hasard parmi les individus non-élites de la population,
        #puis les parents correspondants sont extraits de la population.
        parent1_idx = np.random.randint(num_elites,population_size,size=population_size-num_elites)
        parent2_idx = np.random.randint(num_elites,population_size,size=population_size-num_elites)
        parent1 = population[parent1_idx]
        parent2 = population[parent2_idx]
        # une matrice de probabilités aléatoires pour déterminer
         #quels éléments doivent être échangés entre les parents lors du croisement.
        crossover_prob = np.random.rand(population_size-num_elites,len(data.columns))
        crossover_mask = crossover_prob <= 0.5
        # effectuer le croisement entre les parents
        #en combinant les éléments de parent1 et parent2 en fonction du masque booléen crossover_mask.
        offspring_crossover = np.where(crossover_mask, parent1, parent2)
        # matrice de probabilités aléatoires pour déterminer quels éléments de la population doivent subir une mutation.
        mutation_prob = np.random.rand(population_size-num_elites, len(data.columns))
        mutation_mask = mutation_prob <= mutation_rate
        mutation_values = np.random.rand(population_size-num_elites, len(data.columns))
        # une matrice de directions aléatoires pour déterminer dans quelle direction
        #chaque mutation doit être appliquée lors de la mutation des individus.
        mutation_direction = np.random.choice([-1, 1], size=(population_size-num_elites, len(data.columns)))
# La variable offspring_mutation contient les individus issus du croisement et de la mutation,
    # prêts à être ajoutés à la génération suivante après avoir vérifié qu'ils ont des poids valides.
        # La variable offspring_mutation contient les individus issus du croisement et de la mutation,
# prêts à être ajoutés à la génération suivante après avoir vérifié qu'ils ont des poids valides.
        offspring_mutation = np.where(mutation_mask, offspring_crossover + mutation_direction * mutation_values, offspring_crossover)

        # Clamper les poids entre 0 et 0.6
        offspring_mutation = np.clip(offspring_mutation, 0.1, 0.6)

        # S'assurer que la somme des poids de chaque individu est égale à 1.
        offspring_mutation = offspring_mutation / (np.sum(offspring_mutation, axis=1)[:, np.newaxis] + 1e-8)

        # combine les individus élites et les individus issus du croisement et de la mutation pour créer la génération
        # suivante de la population.
        population = np.vstack((population[:num_elites], offspring_mutation))

        # calculer la performance de la nouvelle population
        fitness = np.array([fitness_function(individual, data.to_numpy(), risk_free_rate) for individual in population])
    best_idx = np.argmax(fitness)
    best_individual = population[best_idx]

    return best_individual

def generate_weights_table(dtr, num_portfolios, anticipated_returns,
                                             population_size,
                                             num_generations,
                                             mutation_rate,
                                             elitism,
                                             risk_free_rate,tau_range=np.linspace(0.001, 1, 100)):
    filtered_df = dtr
    dtr = filtered_df.pct_change()
    dtr = dtr.dropna()
    weights_BL_normalized = black_litterman_model(dtr.cov(), dtr, anticipated_returns, tau_range)
    weights_black_litterman_df = pd.DataFrame(weights_BL_normalized * 1, index=dtr.columns,
                                              columns=['Poids black litterman'])

    k = genetic_algorithm(dtr, population_size, num_generations, mutation_rate, elitism,
                          risk_free_rate)
    weights_genetic_df = pd.DataFrame(k, index=dtr.columns, columns=['Poids algorithme génétique'])
    weights_table = 0.2  # sets every row in this column to 20%
    weights_df = pd.DataFrame(weights_table, index=dtr.columns, columns=['Poids equipondéré'])
    mean_variance_df = MV(dtr,dtr.mean(), dtr.cov(), num_portfolios)
    weights_table = pd.concat([weights_black_litterman_df['Poids black litterman'], mean_variance_df['allocation'],
                               weights_genetic_df['Poids algorithme génétique'], weights_df['Poids equipondéré']],
                              axis=1) * 100
    weights_table.columns = ['Poids black litterman', 'Poids Moyenne-Variance', 'Poids Algorithm Génétique',
                             'Poids equipondéré']
    weights_table['Différence BL/Poids equi'] = weights_table['Poids black litterman'] - weights_df[
        'Poids equipondéré']

    return weights_table

########################################
def generate_weights_plot(generate_weights_table):
    weights_table_style=generate_weights_table
    N = weights_table_style.shape[0]
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_title('Poids du portefeuille du modèle Black-Litterman vs portefeuille du marché vs la moyenne-variance vs algorithme génétique')
    ax.plot(np.arange(N)+1,weights_table_style['Poids Moyenne-Variance'], '^', c='b', label='Moyenne-variance')
    ax.plot(np.arange(N)+1, weights_table_style['Poids equipondéré'], 'o', c='g', label='Portefeuille du marché')
    ax.plot(np.arange(N)+1, weights_table_style['Poids black litterman'], '*', c='r', markersize=10, label='Black-Litterman')
    ax.plot(np.arange(N)+1, weights_table_style['Poids Algorithm Génétique'], '*', c='y', markersize=10, label='Algorithme génétique')
    ax.vlines(np.arange(N)+1, 0,weights_table_style['Poids Moyenne-Variance'], lw=1)
    ax.vlines(np.arange(N)+1, 0,weights_table_style['Poids equipondéré'], lw=1)
    ax.vlines(np.arange(N)+1, 0, weights_table_style['Poids black litterman'], lw=1)
    ax.vlines(np.arange(N)+1, 0,  weights_table_style['Poids Algorithm Génétique'], lw=1)
    ax.axhline(0, c='m')
    ax.axhline(-1, c='m', ls='--')
    ax.axhline(1, c='m', ls='--')
    ax.set_xlabel('Actifs')
    ax.set_ylabel('Poids du portefeuille')
    ax.xaxis.set_ticks(np.arange(1, N+1, 1))
    ax.set_xticklabels(weights_table_style.index.values)
    plt.xticks(rotation=90)
    plt.legend(numpoints=1, fontsize=11)

    # Save the plot as BytesIO object
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Convert the image to base64 encoded string
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return image_base64


# Fonction pour calculer les rendements, les volatilités et les ratios de Sharpe des portefeuilles
def portfolio_performance(weights, returns, cov_matrix, risk_free_rate, data_returns):
    portfolio_return = np.dot(weights, returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

 # Calmar Ratio
    portfolio_returns = np.dot(data_returns, weights)
    portfolio_cum_returns = np.cumprod(1 + portfolio_returns) - 1
    max_drawdown = np.max(np.maximum.accumulate(portfolio_cum_returns) - portfolio_cum_returns)
    calmar_ratio = portfolio_return / max_drawdown

    # Treynor Ratio
    portfolio_returns = np.dot(data_returns, weights)
    portfolio_beta = np.cov(portfolio_returns, data_returns.iloc[:, 0])[0][1] / np.var(data_returns.iloc[:, 0])
    treynor_ratio = (portfolio_return - risk_free_rate) / (portfolio_beta + 1e-8)

    return portfolio_return, portfolio_volatility, sharpe_ratio, calmar_ratio, treynor_ratio


def perfermance(generate_weights_table, dtr, risk_free_rate):
    weights_table_style = generate_weights_table
    data = dtr
    dtr = data.pct_change()
    dtr = dtr.dropna()

    # Calculer les performances des portefeuilles pour chaque approche
    perf_BL = portfolio_performance(weights_table_style['Poids black litterman'] / 100, dtr.mean()*252, dtr.cov(),
                                    risk_free_rate, dtr)
    perf_market = portfolio_performance(weights_table_style['Poids equipondéré'] / 100, dtr.mean()*252, dtr.cov(),
                                        risk_free_rate, dtr)
    perf_MV = portfolio_performance(weights_table_style['Poids Moyenne-Variance'] / 100, dtr.mean()*252, dtr.cov(),
                                    risk_free_rate, dtr)
    perf_gen = portfolio_performance(weights_table_style['Poids Algorithm Génétique'] / 100, dtr.mean()*252, dtr.cov(),
                                     risk_free_rate, dtr)

    # Créer un DataFrame pour afficher les résultats
    results_df = pd.DataFrame([perf_BL, perf_market, perf_MV, perf_gen],
                              columns=['Rendement', 'Volatilité', 'Ratio de Sharpe', 'Calmar Ratio', 'Treynor Ratio'],
                              index=['Black-Litterman', 'Capitalisation Boursière', 'Moyenne-Variance',
                                     'Algorithme génétique'])

    return results_df.T


def calculate_max_drawdown(portfolio_values):
    peak = portfolio_values.max()
    trough = portfolio_values[portfolio_values.idxmax():].min()
    max_drawdown = (trough - peak) / peak
    return max_drawdown


# Include Maximum Drawdown calculation in backtest_markowitz_portfolio function
def backtest_markowitz_portfolio(prices):
    portfolio_value = 1000
    portfolio_values = pd.DataFrame(index=prices.index, columns=['Portfolio'])
    daily_returns = []

    # Calculate mean returns and covariance matrix based on data up to first day
    mean_returns = expected_returns.mean_historical_return(prices.iloc[:2])
    cov_matrix =  risk_models.CovarianceShrinkage(prices.iloc[:2]).ledoit_wolf()

    # Calculate the weights for the first day
    result = max_sharpe_ratio(mean_returns, cov_matrix)
    weights = result.x

    for i in range(1, len(prices)):
        # Calculate the daily returns
        returns = np.dot(prices.iloc[i] - prices.iloc[i - 1], weights)
        portfolio_value += returns
        daily_return = returns / portfolio_value
        daily_returns.append(daily_return)

        # Update the portfolio values dataframe
        portfolio_values['Portfolio'].iloc[i] = portfolio_value

        # Recalculate weights at the beginning of each month (approx every 21 days)
        if i % 21 == 0:
            mean_returns = expected_returns.mean_historical_return(prices.iloc[:i])
            cov_matrix = risk_models.sample_cov(prices.iloc[:i])
            result = max_sharpe_ratio(mean_returns, cov_matrix)
            weights = result.x

    portfolio_values['Portfolio'] = portfolio_values['Portfolio'].fillna(method='bfill')

    # Calculate the cumulative returns
    cumulative_returns = (portfolio_values / portfolio_values.iloc[1]) - 1

    # Calculate the net profit
    net_profit = portfolio_value - portfolio_values['Portfolio'].iloc[1]

    # Calculate the Maximum Drawdown
    max_drawdown = calculate_max_drawdown(portfolio_values['Portfolio'])

    # Plotting the cumulative returns
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_values.index, y=cumulative_returns['Portfolio'] * 100, mode='markers',
                             name='Portfolio Cumulative Return %'))

    for asset in prices.columns:
        asset_cumulative_returns = (prices[asset] / prices[asset].iloc[0]) - 1
        fig.add_trace(go.Scatter(x=portfolio_values.index, y=asset_cumulative_returns * 100, mode='lines', name=asset))

    fig.update_layout(title="Mean Variance Cumulative Returns")



    # Save the plot as BytesIO object with the specified width and height
    buffer = io.BytesIO()
    fig.write_image(buffer, format='png')

    # Convert the image to base64 encoded string
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Calculate the Sharpe ratio and total trades
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns)
    total_trades = len(prices)

    # Create a summary table
    summary = pd.DataFrame({
        'Total Net Profit': [net_profit],
        'Sharpe Ratio': [sharpe_ratio],
        'Total Trades': [total_trades],
        'Maximum Drawdown': [max_drawdown]  # Include Maximum Drawdown
    })

    return summary, image_base64



def backtest_markowitz_portfolio_img(dtr):
    prices = dtr
    summary, image_base64 = backtest_markowitz_portfolio(prices)
    return image_base64



def backtest_black_litterman_portfolios(prices, anticipated_returns):
    portfolio_value = 1000

    portfolio_values = pd.DataFrame(index=prices.index, columns=['Portfolio'])
    daily_returns = []
    returnss = prices.pct_change()
    returnss = returnss.dropna()
    mean_returns = returnss
    cov_matrix = returnss.cov()
    weights = black_litterman_model(cov_matrix, mean_returns, anticipated_returns)
    for i in range(1, len(prices)):

        # Calculate the daily returns
        returns = np.dot(prices.iloc[i] - prices.iloc[i - 1], weights)
        portfolio_value += returns
        daily_return = returns / portfolio_value
        daily_returns.append(daily_return)

        # Update the portfolio values dataframe
        portfolio_values['Portfolio'].iloc[i] = portfolio_value

        # Recalculate weights at the beginning of each month (approx every 21 days)
        if i % 21 == 0:
            mean_returns = returnss.iloc[:i]
            cov_matrix = returnss.iloc[:i].cov()
            weights = black_litterman_model(cov_matrix, mean_returns, anticipated_returns,
                                            tau_range=np.linspace(0.001, 1, 100))

    portfolio_values['Portfolio'] = portfolio_values['Portfolio'].fillna(method='bfill')

    # Calculate the cumulative returns
    cumulative_returns = (portfolio_values / portfolio_values.iloc[0]) - 1

    # Calculate the net profit
    net_profit = portfolio_value - portfolio_values['Portfolio'].iloc[0]

    # Calculate the Maximum Drawdown
    max_drawdown = calculate_max_drawdown(portfolio_values['Portfolio'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_values.index, y=cumulative_returns['Portfolio'] * 100, mode='markers',
                             name='Portfolio Cumulative Return %'))

    for asset in prices.columns:
        asset_cumulative_returns = (prices[asset] / prices[asset].iloc[0]) - 1
        fig.add_trace(go.Scatter(x=portfolio_values.index, y=asset_cumulative_returns * 100, mode='lines', name=asset))

    fig.update_layout(title="Black Litterman Cumulative Returns")


    # Save the plot as BytesIO object with the specified width and height
    buffer = io.BytesIO()
    fig.write_image(buffer, format='png')

    # Convert the image to base64 encoded string
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Calculate the Sharpe ratio and total trades
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns)
    total_trades = len(prices)

    # Create a summary table
    summary = pd.DataFrame({
        'Total Net Profit': [net_profit],
        'Sharpe Ratio': [sharpe_ratio],
        'Total Trades': [total_trades],
        'Maximum Drawdown': [max_drawdown]  # Include Maximum Drawdown
    })

    return summary, image_base64


def backtest_black_litterman_portfolios_imge(dtr, anticipated_returns):
    prices = dtr
    summary, image_base64 = backtest_black_litterman_portfolios(prices, anticipated_returns)
    return image_base64


def backtest_genetic_portfolio(prices, population_size=500, num_generations=100, mutation_rate=0.1, elitism=0.2,
                               risk_free_rate=0.01):
    portfolio_value = 1000
    portfolio_values = pd.DataFrame(index=prices.index, columns=['Portfolio'])
    daily_returns = []
    data_returns = prices.pct_change()
    data_returns = data_returns.dropna()

    # Define a month in terms of trading days
    month_length = 21

    # Initialize weights with genetic algorithm
    weights = genetic_algorithm(data_returns.iloc[:month_length], population_size, num_generations, mutation_rate,
                                elitism, risk_free_rate)

    for i in range(month_length, len(prices)):
        # If the current day is the start of a new month, update the weights
        if (i - month_length) % month_length == 0:
            weights = genetic_algorithm(data_returns.iloc[i - month_length + 1:i + 1], population_size, num_generations,
                                        mutation_rate, elitism, risk_free_rate)

        # Calculate the daily returns
        returns = np.dot(prices.iloc[i] - prices.iloc[i - 1], weights)
        portfolio_value += returns
        daily_return = returns / portfolio_value
        daily_returns.append(daily_return)

        # Update the portfolio values dataframe
        portfolio_values['Portfolio'].iloc[i] = portfolio_value

    portfolio_values['Portfolio'] = portfolio_values['Portfolio'].fillna(method='bfill')

    # Calculate the cumulative returns
    cumulative_returns = (portfolio_values / portfolio_values.iloc[0]) - 1

    # Calculate the net profit
    net_profit = portfolio_value - portfolio_values['Portfolio'].iloc[0]

    # Calculate the Maximum Drawdown
    max_drawdown = calculate_max_drawdown(portfolio_values['Portfolio'])

    # Plotting the cumulative returns
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_values.index, y=cumulative_returns['Portfolio'] * 100, mode='markers',
                             name='Portfolio Cumulative Return %'))

    for asset in prices.columns:
        asset_cumulative_returns = (prices[asset] / prices[asset].iloc[0]) - 1
        fig.add_trace(go.Scatter(x=portfolio_values.index, y=asset_cumulative_returns * 100, mode='lines', name=asset))

    fig.update_layout(title="Genetic Algorithm Cumulative Returns")



    # Save the plot as BytesIO object with the specified width and height
    buffer = io.BytesIO()
    fig.write_image(buffer, format='png')

    # Convert the image to base64 encoded string
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Calculate the Sharpe ratio and total trades
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns)
    total_trades = len(prices)

    # Create a summary table
    summary = pd.DataFrame({
        'Total Net Profit': [net_profit],
        'Sharpe Ratio': [sharpe_ratio],
        'Total Trades': [total_trades],
        'Maximum Drawdown': [max_drawdown]  # Include Maximum Drawdown
    })

    return summary, image_base64


def backtest_genetic_portfolio_image(dtr, population_size, num_generations, mutation_rate, elitism, risk_free_rate):
    prices = dtr
    summary, image_base64 = backtest_genetic_portfolio(prices, population_size, num_generations, mutation_rate, elitism,
                                                       risk_free_rate)

    return image_base64
def run_backtests(dtr, anticipated_returns, population_size, num_generations, mutation_rate, elitism, risk_free_rate):
    # Run the backtest functions
    prices=dtr
    s_markowitz,_= backtest_markowitz_portfolio(prices)
    s_markowitz.index = ['Markowitz']

    s_black_litterman,_ = backtest_black_litterman_portfolios(prices, anticipated_returns)
    s_black_litterman.index = ['Black-Litterman']

    s_genetic,_ = backtest_genetic_portfolio(prices, population_size, num_generations, mutation_rate, elitism, risk_free_rate)
    s_genetic.index = ['Genetic Algorithm']

    # Concatenate the results
    results = pd.concat([s_markowitz, s_black_litterman, s_genetic])

    return results
