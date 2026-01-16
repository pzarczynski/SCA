# S-box

Funkcja S-box to pierwsza nieliniowa operacja w algorytmie AES. Polega ona na podmianie bajtu tekstu jawnego $d$ zmieszanego z bajtem klucza $k$ przy użyciu [tabeli podstawień](https://csrc.nist.gov/files/pubs/fips/197/final/docs/fips-197.pdf#page=20):

$$
z = \mathrm{Sbox}(d \oplus k)
$$

Operacja S-box jest wrażliwa, bo jest to punkt, w którym klucz jest po raz pierwszy bezpośrednio mieszany z danymi wejściowymi. Nieliniowość tej operacji ułatwia statystyczne odróżnienie poprawnego klucza od błędnego.
