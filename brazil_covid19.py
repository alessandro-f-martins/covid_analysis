# %matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
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


# Gera as variações diárias de casos e mortes
delta_cases = [int(covid_SP.cases.iloc[i+1] - covid_SP.cases.iloc[i])
               for i in range(covid_SP.cases.size - 1)]
delta_cases.insert(0, 0)
delta_deaths = [int(covid_SP.cases.iloc[i+1] - covid_SP.cases.iloc[i])
                for i in range(covid_SP.cases.size - 1)]
delta_deaths.insert(0, 0)
covid_SP['delta_cases'] = delta_cases
covid_SP['delta_deaths'] = delta_deaths


# Gera as funções de projeção
day_number = np.array(range(covid_SP.cases.size))

popt_cases, pcov_cases = curve_fit(fit_func_cases, day_number,
                                   covid_SP.cases.values)
projected_cases = [fit_func_cases(d, *popt_cases) for d in day_number]
popt_dcases, pcov_dcases = curve_fit(fit_func_dcases, day_number,
                                     covid_SP['delta_cases'].values)
projected_dcases = [fit_func_dcases(d, *popt_dcases) for d in day_number]

covid_SP['projected_cases'] = projected_cases
covid_SP['projected_dcases'] = projected_dcases

# Cria os gráficos
plt.plot_date(x=covid_SP.index, y=covid_SP.cases,
              label='Casos', marker=',', ls='-')
plt.plot_date(x=covid_SP.index, y=covid_SP['delta_cases'],
              label='Variação diária dos casos', marker=',', ls='-')
plt.plot_date(x=covid_SP.index, y=covid_SP.cases,
              label='Mortes', marker=',', ls='-')
plt.plot_date(x=covid_SP.index, y=covid_SP['delta_deaths'],
              label='Variação diária das mortes', marker=',', ls='-')
plt.plot_date(x=covid_SP.index, y=covid_SP['projected_cases'],
              label='Casos projetados', marker=',', ls='--')
plt.plot_date(x=covid_SP.index, y=covid_SP['projected_dcases'],
              label='Variação diária dos casos (projetado)',
              marker=',', ls='--')

plt.title('Casos de Covid-19 no Estado de São Paulo até %s' %
          datetime.strftime(covid_SP.index[-1], '%d/%m/%Y'))
plt.legend()
plt.show()