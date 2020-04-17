# -*- coding: utf-8 -*-

# %matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
import fit_funcs


fit_func_cases = fit_funcs.fourth_func
fit_func_dcases = fit_funcs.sixth_func

# Le os casos do arquivo CSV:
covid = pd.read_csv('./brazil_covid19.csv')
covid['date'] = pd.to_datetime(covid['date'])
temp = covid[covid['state'] == 'São Paulo'].drop(
    columns=['state', 'region']).reset_index(drop=True)
covid_SP = pd.DataFrame(data=temp.drop(columns='date').values,
                        index=temp['date'].values,
                        columns=['cases', 'deaths'])

today = covid_SP.index[-1]

# Gera as variações diárias de casos e mortes:
delta_cases = [int(covid_SP.cases.iloc[i+1] - covid_SP.cases.iloc[i])
               for i in range(covid_SP.cases.size - 1)]
delta_cases.insert(0, 0)
delta_deaths = [int(covid_SP.deaths.iloc[i+1] - covid_SP.deaths.iloc[i])
                for i in range(covid_SP.deaths.size - 1)]
delta_deaths.insert(0, 0)
covid_SP['delta_cases'] = delta_cases
covid_SP['delta_deaths'] = delta_deaths


# Gera as funções de projeção:
day_number = np.array(range(covid_SP.cases.size))

popt_cases, _ = curve_fit(fit_func_cases, day_number,
                          covid_SP.cases.values)
projected_cases = [fit_func_cases(d, *popt_cases) for d in day_number]

popt_dcases, _ = curve_fit(fit_func_dcases, day_number,
                           covid_SP.delta_cases.values)
projected_dcases = [fit_func_dcases(d, *popt_dcases) for d in day_number]

covid_SP['projected_cases'] = projected_cases
covid_SP['projected_dcases'] = projected_dcases

# Cria a projeção para o próximo mês:
added_days = np.array(range(day_number[-1] + 1, day_number[-1] + 31))
added_cases = [fit_func_cases(d, *popt_cases) for d in added_days]
added_dcases = [fit_func_dcases(d, *popt_dcases) for d in added_days]

for added_case, added_dcase in zip(added_cases, added_dcases):
    covid_add = pd.DataFrame([[added_case, added_dcase]],
                            columns=['projected_cases', 'projected_dcases'],
                            index=[covid_SP.index[-1] +
                                   timedelta(days=1)])
    covid_SP = covid_SP.append(covid_add)           

# Cria os gráficos:
plt.plot_date(x=covid_SP.index, y=covid_SP.cases,
              label='Casos', marker=',', ls='-')
plt.plot_date(x=covid_SP.index, y=covid_SP['delta_cases'],
              label='Variação diária dos casos', marker=',', ls='-')
plt.plot_date(x=covid_SP.index, y=covid_SP.deaths,
              label='Mortes', marker=',', ls='-')
plt.plot_date(x=covid_SP.index, y=covid_SP['delta_deaths'],
              label='Variação diária das mortes', marker=',', ls='-')
plt.plot_date(x=covid_SP.index, y=covid_SP['projected_cases'],
              label='Casos projetados', marker=',', ls='--')
plt.plot_date(x=covid_SP.index, y=covid_SP['projected_dcases'],
              label='Variação diária dos casos (projetado)',
              marker=',', ls='--')
plt.axvline(x=today, ls=':', c='m',
            label='Hoje (%s)' % datetime.strftime(today, '%d/%m/%Y'))

plt.title('Casos de Covid-19 no Estado de São Paulo até %s' %
          datetime.strftime(covid_SP.index[-1], '%d/%m/%Y'))
plt.legend()
plt.show()
