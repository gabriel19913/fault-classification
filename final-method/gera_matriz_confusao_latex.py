from bs4 import BeautifulSoup
import numpy as np

model, den, method, cycle, signal, name, nome_cap = 'cm', '128', 'G', '1/128', 'i', 'minirocket', 'MiniRocket'
n1, n2 = 38, 36

with open(f'./figs_cm/new_dataset/cycle_{den}_{name}.svg', 'rb') as f:
    svg = f.read()

soup = BeautifulSoup(svg, 'html.parser')
values = soup.find_all(class_="annotation")[:-2]
values = [int(v.text) for v in values]
ct_ = values[0:10]
ct_ = [str(v) if v != 0 else '' for v in ct_]
ct_ = [v + '/' + str(n1) if v != '' else v for v in ct_]
ct_ = [v + r"\end{tabular} " if v != '' else v for v in ct_]
ct = list(np.round((np.array(values)[0:10] / n1) * 100, 2))
ct = [str(v) if v != 0 else '' for v in ct]
ct = [v.replace('.', ',') for v in ct]
ct = [r" \begin{tabular}[c]{@{}c@{}}" + v + r"\%\\ " if v != '' else v for v in ct]

cat_ = values[10:20]
cat_ = [str(v) if v != 0 else '' for v in cat_]
cat_ = [v + '/' + str(n2) if v != '' else v for v in cat_]
cat_ = [v + r"\end{tabular} " if v != '' else v for v in cat_]
cat = list(np.round((np.array(values)[10:20] / n2) * 100, 2))
cat = [str(v) if v != 0 else '' for v in cat]
cat = [v.replace('.', ',') for v in cat]
cat = [r" \begin{tabular}[c]{@{}c@{}}" + v + r"\%\\ " if v != '' else v for v in cat]

ca_ = values[20:30]
ca_ = [str(v) if v != 0 else '' for v in ca_]
ca_ = [v + '/' + str(n1) if v != '' else v for v in ca_]
ca_ = [v + r"\end{tabular} " if v != '' else v for v in ca_]
ca = list(np.round((np.array(values)[20:30] / n1) * 100, 2))
ca = [str(v) if v != 0 else '' for v in ca]
ca = [v.replace('.', ',') for v in ca]
ca = [r" \begin{tabular}[c]{@{}c@{}}" + v + r"\%\\ " if v != '' else v for v in ca]

bt_ = values[30:40]
bt_ = [str(v) if v != 0 else '' for v in bt_]
bt_ = [v + '/' + str(n1) if v != '' else v for v in bt_]
bt_ = [v + r"\end{tabular} " if v != '' else v for v in bt_]
bt = list(np.round((np.array(values)[30:40] / n1) * 100, 2))
bt = [str(v) if v != 0 else '' for v in bt]
bt = [v.replace('.', ',') for v in bt]
bt = [r" \begin{tabular}[c]{@{}c@{}}" + v + r"\%\\ " if v != '' else v for v in bt]

bct_ = values[40:50]
bct_ = [str(v) if v != 0 else '' for v in bct_]
bct_ = [v + '/' + str(n1) if v != '' else v for v in bct_]
bct_ = [v + r"\end{tabular} " if v != '' else v for v in bct_]
bct = list(np.round((np.array(values)[40:50] / n1) * 100, 2))
bct = [str(v) if v != 0 else '' for v in bct]
bct = [v.replace('.', ',') for v in bct]
bct = [r" \begin{tabular}[c]{@{}c@{}}" + v + r"\%\\ " if v != '' else v for v in bct]

bc_ = values[50:60]
bc_ = [str(v) if v != 0 else '' for v in bc_]
bc_ = [v + '/' + str(n1) if v != '' else v for v in bc_]
bc_ = [v + r"\end{tabular} " if v != '' else v for v in bc_]
bc = list(np.round((np.array(values)[50:60] / n1) * 100, 2))
bc = [str(v) if v != 0 else '' for v in bc]
bc = [v.replace('.', ',') for v in bc]
bc = [r" \begin{tabular}[c]{@{}c@{}}" + v + r"\%\\ " if v != '' else v for v in bc]

at_ = values[60:70]
at_ = [str(v) if v != 0 else '' for v in at_]
at_ = [v + '/' + str(n1) if v != '' else v for v in at_]
at_ = [v + r"\end{tabular} " if v != '' else v for v in at_]
at = list(np.round((np.array(values)[60:70] / n1) * 100, 2))
at = [str(v) if v != 0 else '' for v in at]
at = [v.replace('.', ',') for v in at]
at = [r" \begin{tabular}[c]{@{}c@{}}" + v + r"\%\\ " if v != '' else v for v in at]

abt_ = values[70:80]
abt_ = [str(v) if v != 0 else '' for v in abt_]
abt_ = [v + '/' + str(n1) if v != '' else v for v in abt_]
abt_ = [v + r"\end{tabular} " if v != '' else v for v in abt_]
abt = list(np.round((np.array(values)[70:80] / n1) * 100, 2))
abt = [str(v) if v != 0 else '' for v in abt]
abt = [v.replace('.', ',') for v in abt]
abt = [r" \begin{tabular}[c]{@{}c@{}}" + v + r"\%\\ " if v != '' else v for v in abt]

abc_ = values[80:90]
abc_ = [str(v) if v != 0 else '' for v in abc_]
abc_ = [v + '/' + str(n2) if v != '' else v for v in abc_]
abc_ = [v + r"\end{tabular} " if v != '' else v for v in abc_]
abc = list(np.round((np.array(values)[80:90] / n2) * 100, 2))
abc = [str(v) if v != 0 else '' for v in abc]
abc = [v.replace('.', ',') for v in abc]
abc = [r" \begin{tabular}[c]{@{}c@{}}" + v + r"\%\\ " if v != '' else v for v in abc]

ab_ = values[90:100]
ab_ = [str(v) if v != 0 else '' for v in ab_]
ab_ = [v + '/' + str(n1) if v != '' else v for v in ab_]
ab_ = [v + r"\end{tabular} " if v != '' else v for v in ab_]
ab = list(np.round((np.array(values)[90:100] / n1) * 100, 2))
ab = [str(v) if v != 0 else '' for v in ab]
ab = [v.replace('.', ',') for v in ab]
ab = [r" \begin{tabular}[c]{@{}c@{}}" + v + r"\%\\ " if v != '' else v for v in ab]

s = rf'''
A Figura~\ref{{{model}_{den}_ciclo_{method}_{name}}} apresenta a matriz de confusão gerada pelo método {method} treinado com sinais com {cycle} ciclo pós falta e extração de \textit{{features}} usando \textit{{{nome_cap}}}.

\begin{{table}}[H]
    \centering
    \caption{{Matriz de confusão do método {method} com {cycle} ciclo pós falta e extração de características com \textit{{{nome_cap}}}.}}
    \label{{{model}_{den}_ciclo_{method}_{name}}}
    \begin{{adjustbox}}{{width=0.8\columnwidth,center}}
    \renewcommand{{\arraystretch}}{{1.5}}
    \begin{{tabular}}{{cccccccccccc}}
    \hline
    \multicolumn{{1}}{{l}}{{}} & \multicolumn{{11}}{{c}}{{\textbf{{Valores preditos}}}} \\ \hline
    \multirow{{11}}{{*}}{{\rotatebox[origin=c]{{90}}{{\textbf{{Valores reais}}}}}} &  & AB & ABC & ABT & AT & BC & BCT & BT & CA & CAT & CT \\ \cline{{2-12}} 
     & AB &{ab[0]}{ab_[0]}&{ab[1]}{ab_[1]}&{ab[2]}{ab_[2]}&{ab[3]}{ab_[3]}&{ab[4]}{ab_[4]}&{ab[5]}{ab_[5]}&{ab[6]}{ab_[6]}&{ab[7]}{ab_[7]}&{ab[8]}{ab_[8]}&{ab[9]}{ab_[9]}\\ \cline{{2-12}} 
     & ABC &{abc[0]}{abc_[0]}&{abc[1]}{abc_[1]}&{abc[2]}{abc_[2]}&{abc[3]}{abc_[3]}&{abc[4]}{abc_[4]}&{abc[5]}{abc_[5]}&{abc[6]}{abc_[6]}&{abc[7]}{abc_[7]}&{abc[8]}{abc_[8]}&{abc[9]}{abc_[9]}\\ \cline{{2-12}} 
     & ABT &{abt[0]}{abt_[0]}&{abt[1]}{abt_[1]}&{abt[2]}{abt_[2]}&{abt[3]}{abt_[3]}&{abt[4]}{abt_[4]}&{abt[5]}{abt_[5]}&{abt[6]}{abt_[6]}&{abt[7]}{abt_[7]}&{abt[8]}{abt_[8]}&{abt[9]}{abt_[9]}\\ \cline{{2-12}} 
     & AT &{at[0]}{at_[0]}&{at[1]}{at_[1]}&{at[2]}{at_[2]}&{at[3]}{at_[3]}&{at[4]}{at_[4]}&{at[5]}{at_[5]}&{at[6]}{at_[6]}&{at[7]}{at_[7]}&{at[8]}{at_[8]}&{at[9]}{at_[9]}\\ \cline{{2-12}} 
     & BC &{bc[0]}{bc_[0]}&{bc[1]}{bc_[1]}&{bc[2]}{bc_[2]}&{bc[3]}{bc_[3]}&{bc[4]}{bc_[4]}&{bc[5]}{bc_[5]}&{bc[6]}{bc_[6]}&{bc[7]}{bc_[7]}&{bc[8]}{bc_[8]}&{bc[9]}{bc_[9]}\\ \cline{{2-12}} 
     & BCT &{bct[0]}{bct_[0]}&{bct[1]}{bct_[1]}&{bct[2]}{bct_[2]}&{bct[3]}{bct_[3]}&{bct[4]}{bct_[4]}&{bct[5]}{bct_[5]}&{bct[6]}{bct_[6]}&{bct[7]}{bct_[7]}&{bct[8]}{bct_[8]}&{bct[9]}{bct_[9]}\\ \cline{{2-12}} 
     & BT &{bt[0]}{bt_[0]}&{bt[1]}{bt_[1]}&{bt[2]}{bt_[2]}&{bt[3]}{bt_[3]}&{bt[4]}{bt_[4]}&{bt[5]}{bt_[5]}&{bt[6]}{bt_[6]}&{bt[7]}{bt_[7]}&{bt[8]}{bt_[8]}&{bt[9]}{bt_[9]}\\ \cline{{2-12}} 
     & CA &{ca[0]}{ca_[0]}&{ca[1]}{ca_[1]}&{ca[2]}{ca_[2]}&{ca[3]}{ca_[3]}&{ca[4]}{ca_[4]}&{ca[5]}{ca_[5]}&{ca[6]}{ca_[6]}&{ca[7]}{ca_[7]}&{ca[8]}{ca_[8]}&{ca[9]}{ca_[9]}\\ \cline{{2-12}} 
     & CAT &{cat[0]}{cat_[0]}&{cat[1]}{cat_[1]}&{cat[2]}{cat_[2]}&{cat[3]}{cat_[3]}&{cat[4]}{cat_[4]}&{cat[5]}{cat_[5]}&{cat[6]}{cat_[6]}&{cat[7]}{cat_[7]}&{cat[8]}{cat_[8]}&{cat[9]}{cat_[9]}\\ \cline{{2-12}} 
     & CT &{ct[0]}{ct_[0]}&{ct[1]}{ct_[1]}&{ct[2]}{ct_[2]}&{ct[3]}{ct_[3]}&{ct[4]}{ct_[4]}&{ct[5]}{ct_[5]}&{ct[6]}{ct_[6]}&{ct[7]}{ct_[7]}&{ct[8]}{ct_[8]}&{ct[9]}{ct_[9]}\\ \hline
    \end{{tabular}}
    \end{{adjustbox}}
    {{\newline \\ \small Do autor (2021).}}
    \end{{table}}
'''

print(s)
