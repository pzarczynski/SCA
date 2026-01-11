# Side Channel Analysis

## Problem badawczy

Czy techniki uczenia maszynowego są w stanie przełamać zabezpieczenie typu *masking* w **SCA**?

**SCA** (*Side Channel Analysis*) to klasa ataków kryptoanalitycznych, która ignoruje matematyczną siłę algorytmu, a zamiast tego atakuje jego fizyczną implementację.

### Dane

W projekcie wykorzystuję zbiór danych [ASCAD v1 variable key](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_variable_key), zawierający próbki pochodzące z mikrokontrolera ATMega8515 wykonującego szyfrowanie algorytmem AES.

### Cel

Celem jest pozyskanie klucza, którego kontroler używa do szyfrowania danych, na podstawie zmierzonych wartości pola.
Teoretycznie jest to możliwe - pole elektromagnetyczne emitowane przez mikrokontroler jest skorelowane z wagą Hamminga przetwarzanych danych.

Zadanie to jest klasyfikacją szeregów czasowych, o tyle specyficzną, że poszczególne timestampy możemy traktować również jako cechy, ponieważ ślady są ze sobą zsynchronizowane.

#### [Więcej o danych](reports/ASCAD_v1.md)

### Akumulacja wiarygodności

Problemem w SCA jest bardzo duże zaszumienie danych, przez co ciężko uzyskać pewny model. Jak przeglądałem literaturę związaną z tym zagadnieniem, zauważyłem, że model jest tu traktowany jako swojego rodzaju weak learner, a cała siła ataku tkwi w akumulacji wiarygodności na przestrzeni ataku. Co to znaczy? Załóżmy, że mamy $N$ próbek walidacyjnych $T$ pochodzących z szyfrowania tym samym kluczem. Ostateczne prawdopodobieństwo, że dany kandydat na etykietę $\hat{z}$ jest poprawny to:
$$log(P(\hat{z} | T)) = \sum_{i=0}^{N}{log(P(\hat{z} | T_i))} $$

### Ewaluacja

Metryka GuessingEntropy mierzy oczekiwaną liczbę zgadywań, jakie atakujący musi wykonać, testując klucze w kolejności od najbardziej do najmniej prawdopodobnego według modelu.
Do ewaluacji modelu będę wykorzystywać MeanGE, czyli średnią z GE dla każdego momentu akumulacji podczas ataku.

$$MeanGE(k, T^{(z)}) = \frac{1}{\hat{Z}}\sum_{i=0}^{\hat{Z}}{GE(z, P(z|T_{0...i}^{(z)}))}$$

$T^{(z)}$ - ślady z etykietą $z$; $T_{0...i}$ ślady od 0 do $i$.

Można zauważyć, że metryka ta jest uzależniona od kolejności śladów $T^{(z)}$, więc dla pewności można wykonać $L$ eksperymentów z różnie potasowanymi śladami, żeby upewnić się, że niski score nie wynika z korzystnego ułożenia danych.

W przeciwieństwie do Accuracy pozostaje sensowna nawet wtedy, gdy poprawny label $z$ rzadko jest na pozycji 1, ale zwykle pojawia się wysoko w rankingu. Ostateczny score:

$$score = \frac{1}{Z}\sum_{i=0}^{Z}{MeanGE(z_i, T^{(z_i)})}$$

Gdzie $Z$ - liczba unikalnych etykiet w zbiorze walidacyjnym; $T_{z_i}$ - ślady dla których etykieta to $z_i$.
