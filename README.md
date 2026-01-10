# Side Channel Analysis

## Problem badawczy

Czy techniki uczenia maszynowego są w stanie przełamać zabezpieczenie typu *masking* w **SCA**?

**SCA** (*Side Channel Analysis*) to klasa ataków kryptoanalitycznych, która ignoruje matematyczną siłę algorytmu, a zamiast tego atakuje jego fizyczną implementację.

### Dane

W projekcie wykorzystuję zbiór danych [`ASCADv1_fixed`](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1), zawierający próbki pochodzące z mikrokontrolera ATMega8515 wykonującego szyfrowanie algorytmem AES.

### Cel

Celem jest pozyskanie klucza, którego kontroler używa do szyfrowania danych, na podstawie zmierzonych wartości pola.

Teoretycznie jest to możliwe - pole elektromagnetyczne emitowane przez mikrokontroler jest skorelowane z wagą Hamminga przetwarzanych danych.

### Atak

W modelu ataku zakłada się, że każdy pomiar jest niezależny. Całkowite prawdopodobieństwo (wiarygodność) dla danego kandydata na klucz $k \in \{0, 1, \dots, 255\}$ jest iloczynem prawdopodobieństw dla każdego zarejestrowanego śladu $t_i$

$$
P(T|k) = \prod_{i=1}^N P(t_i|k)
$$

### Ewaluacja

Jako główny wyznacznik jakości danego modelu, będę używać metryki $\mathrm{MeanRank}$. Mając $N$ próbek pochodzących z szyfrowania tym samym kluczem,  