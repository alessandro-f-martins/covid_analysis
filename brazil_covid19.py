# -*- coding: utf-8 -*-

# %matplotlib

from json import load as j_load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
from fit_funcs import fit_funcs

# STATES = {'Acre': 'o', 'Alagoas': 'e', 'Amapá':'o', 'Amazonas':'o',
#           'Bahia': 'a', 'Ceará':'o', 'Distrito Federal': 'o',
#           'Espírito Santo':'o', 'Goiás':'e', 'Maranhão':'o',
#           'Mato Grosso':'o', 'Mato Grosso do Sul':'o', 'Minas Gerais':'e',
#           'Paraná':'o', 'Paraíba':'a', 'Pará':'o', 'Pernambuco':'o',
#           'Piauí':'o', 'Rio Grande do Norte':'o',
#           'Rio Grande do Sul':'o', 'Rio de Janeiro':'o', 'Rondônia':'e',
#           'Roraima':'e', 'Santa Catarina':'e', 'Sergipe':'e',
#           'São Paulo':'e', 'Tocantins':'o'}

with (open('./covid_config.json', 'r')) as cfg_file:
    cfg = j_load(cfg_file)


# Le os casos do arquivo CSV:
def get_df():
    df = pd.read_csv(cfg['INPUT_FILE'], sep=cfg['INPUT_FILE_SEP'])
    df[cfg['DATE_COL']] = pd.to_datetime(df[cfg['DATE_COL']])
    states = np.unique(df[cfg['STATE_COL']].values)

    return df, states


# Seleciona o dado por Estado
def get_state_data(df, state='São Paulo'):
    temp = df[df[cfg['STATE_COL']] == state][
        [cfg['DATE_COL']] + cfg['DATA_COLS']].reset_index(drop=True)
    ret_df = pd.DataFrame(data=temp.drop(columns=cfg['DATE_COL']).values,
                          index=temp[cfg['DATE_COL']].values,
                          columns=cfg['DATA_COLS'])
    today = ret_df.index[-1]

    return ret_df, today


# Gera as variações diárias de casos e mortes:
def generate_daily_delta(df):
    for col in cfg['COLS_FOR_DELTA']:
        delta_col = [int(df[col].iloc[i+1] - df[col].iloc[i])
                     for i in range(df[col].size - 1)]
        delta_col.insert(0, 0)
        df['delta_' + col] = delta_col

    return df


# Gera as funções de projeção:
def generate_projection_function(df):
    fit_cols = cfg['COLS_FOR_PROJECTION'].copy()
    day_number = np.arange(df[cfg['COLS_FOR_PROJECTION'][0]['name']].size)

    for col in fit_cols:
        coefs, _ = curve_fit(fit_funcs[col['fit_func']], day_number,
                             df[col['name']].values)
        df['projected_' + col['name']] = [fit_funcs[col['fit_func']](d, *coefs)
                                          for d in day_number]
        col['coefs'] = coefs

    return df, fit_cols


# Cria a projeção para o próximo período:
def generate_next_period_projection(df, fit_cols, period=30):
    first_period = df[cfg['COLS_FOR_PROJECTION'][0]['name']].size
    added_days = np.arange(first_period, first_period + period + 1)
    added_data = []
    # cols = tuple(fit_cols.keys())

    for col in fit_cols:
        added_data.append([fit_funcs[col['fit_func']](d, *col['coefs'])
                           for d in added_days])

    for added_d in zip(*added_data):
        covid_add = pd.DataFrame(
            [list(added_d)],
            columns=['projected_' + c['name'] for c in fit_cols],
            index=[df.index[-1] + timedelta(days=1)])
        df = df.append(covid_add)

    return df


# Cria os gráficos:
def plot_graphics(df, today, state, plot_projected=True):
    plt.plot_date(x=df.index, y=df[cfg['DATA_COLS'][0]],
                  label='Casos', marker=',', ls='-')
    plt.plot_date(x=df.index, y=df['delta_' + cfg['COLS_FOR_DELTA'][0]],
                  label='Variação diária dos casos', marker=',', ls='-')
    plt.plot_date(x=df.index, y=df[cfg['DATA_COLS'][1]],
                  label='Mortes', marker=',', ls='-')
    plt.plot_date(x=df.index, y=df['delta_' + cfg['COLS_FOR_DELTA'][1]],
                  label='Variação diária das mortes', marker=',', ls='-')

    _df = df.copy() if plot_projected else df[df.index[0]: today]
    plt.plot_date(x=_df.index,
                  y=_df['projected_' + cfg['COLS_FOR_PROJECTION'][0]['name']],
                  label='Casos projetados', marker=',', ls='--')
    plt.plot_date(x=_df.index,
                  y=_df['projected_' + cfg['COLS_FOR_PROJECTION'][1]['name']],
                  label='Variação diária dos casos (projetado)',
                  marker=',', ls='--')
    plt.axvline(x=today, ls=':', c='m',
                label='Hoje (%s)' % datetime.strftime(today, '%d/%m/%Y'))

    last_day = datetime.strftime(df.index[-1] if plot_projected else today,
                                 '%d/%m/%Y')
    plt.title('Casos de Covid-19 no Estado de %s até %s' % (state, last_day))
    plt.legend()
    plt.show()


if __name__ == "__main__":

    df, states = get_df()

    while True:
        state = input('Entre com o nome do Estado para análise (ou qualquer outra linha para sair): ')
        if state not in states:
            break

        state_df, today = get_state_data(df, state)
        state_df = generate_daily_delta(state_df)
        state_df, fit_cols = generate_projection_function(state_df)

        try:
            use_proj = int(input('Projeção para quantos dias (0 para não gerar): '))
        except ValueError:
            use_proj = 0
        if use_proj:
            state_df = generate_next_period_projection(state_df,
                                                       fit_cols, use_proj)
        plot_graphics(state_df, today, state, use_proj != 0)
        print('Seu gráfico foi gerado!')

    print('Obrigado!')
