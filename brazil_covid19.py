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
    df[cfg['DATE_COL']] = pd.to_datetime(df[cfg['DATE_COL']],
                                         format=cfg['INPUT_TIME_FORMAT'])
    states = np.unique(df[cfg['STATE_COL']].values).tolist()
    states.append('BR')

    return df, states


# Seleciona o dado por Estado
def get_state_data(df, state='São Paulo'):
    if state == 'BR':
        state_df = df.groupby(cfg['DATE_COL']).sum()[cfg['DATA_COLS']]
    else:
        temp = df[df[cfg['STATE_COL']] == state][
            [cfg['DATE_COL']] + cfg['DATA_COLS']
        ].reset_index(drop=True)
        state_df = pd.DataFrame(data=temp.drop(columns=cfg['DATE_COL']).values,
                                index=temp[cfg['DATE_COL']].values,
                                columns=cfg['DATA_COLS'])

    today = state_df.index[-1]
    if cfg['TRIM_INITIAL_ZEROS']:
        state_df = state_df[(state_df.T != 0).any()]

    return state_df, today


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


# Calcula qual funcao produz os minimos quadrados
def calc_least_squares(df):
    fit_cols = cfg['COLS_FOR_PROJECTION'].copy()
    day_number = np.arange(df[cfg['COLS_FOR_PROJECTION'][0]['name']].size)

    for c in fit_cols:
        c['lsq'] = np.inf
        for f in filter(lambda f: f.startswith('poly_'), fit_funcs):
            coefs, _ = curve_fit(fit_funcs[f], day_number,
                                 df[c['name']].values)

            hyp = np.array([fit_funcs[f](d, *coefs) for d in day_number])
            lsq_val = np.sum(np.square(hyp - df[c['name']].values))
            if c['lsq'] > lsq_val:
                c['lsq'] = lsq_val
                c['coefs'] = coefs
                c['fit_func'] = f
                df['projected_' + c['name']] = hyp

    return df, fit_cols


# Cria a projeção para o próximo período:
def generate_next_period_projection(df, fit_cols, period=30):
    first_period = df[cfg['COLS_FOR_PROJECTION'][0]['name']].size
    added_days = np.arange(first_period, first_period + period + 1)
    added_data = []

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
    fig, ax = plt.subplots()
    box_props = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5}

    ax.plot_date(x=df.index, y=df[cfg['DATA_COLS'][0]],
                 label='Casos', marker=',', ls='-')
    ax.plot_date(x=df.index, y=df['delta_' + cfg['COLS_FOR_DELTA'][0]],
                 label='Variação diária dos casos', marker=',', ls='-')
    ax.plot_date(x=df.index, y=df[cfg['DATA_COLS'][1]],
                 label='Mortes', marker=',', ls='-')
    ax.plot_date(x=df.index, y=df['delta_' + cfg['COLS_FOR_DELTA'][1]],
                 label='Variação diária das mortes', marker=',', ls='-')

    ax.text(x=0.6, y=0.9,
            s='Casos hoje: %d' % df.loc[today][cfg['DATA_COLS'][0]],
            transform=ax.transAxes, fontstyle='italic', bbox=box_props)

    _df = df.copy() if plot_projected else df[df.index[0]: today]
    ax.plot_date(x=_df.index,
                 y=_df['projected_' + cfg['COLS_FOR_PROJECTION'][0]['name']],
                 label='Casos projetados', marker=',', ls='--')
    ax.plot_date(x=_df.index,
                 y=_df['projected_' + cfg['COLS_FOR_PROJECTION'][1]['name']],
                 label='Variação diária dos casos (projetado)',
                 marker=',', ls='--')
    ax.axvline(x=today, ls=':', c='m',
               label='Hoje (%s)' % datetime.strftime(today, '%d/%m/%Y'))
    last_day = datetime.strftime(df.index[-1]
                                 if plot_projected else today, '%d/%m/%Y')

    if plot_projected:
        last_value = int(_df['projected_' + cfg['COLS_FOR_PROJECTION']
                             [0]['name']].iloc[-1])
        ax.text(x=0.8, y=0.9,
                s='Casos no último dia\nda projeção: %d' % last_value,
                transform=ax.transAxes, fontstyle='italic', bbox=box_props)

    title = f'Casos de Covid-19 no {"Estado de " + state if state != "BR" else "Brasil"} até {last_day}'
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == "__main__":

    df, states = get_df()

    while True:
        state = input(
            'Nome do Estado para análise (ou qualquer outro texto para sair): '
        )
        if state not in states:
            break

        state_df, today = get_state_data(df, state)
        state_df, fit_cols = generate_projection_function(
            generate_daily_delta(state_df))

        try:
            use_proj = int(
                input('Dias para projeção (0 ou Enter para não gerar): '))
        except ValueError:
            use_proj = 0
        if use_proj:
            state_df = generate_next_period_projection(state_df,
                                                       fit_cols, use_proj)
        print('Numero de casos no ultimo dia da série: %d'
              % int(state_df['projected_' + cfg['COLS_FOR_PROJECTION']
                             [0]['name']].iloc[-1]))
        print('Seu gráfico foi gerado!')
        plot_graphics(state_df, today, state, use_proj != 0)

    print('Obrigado!')
