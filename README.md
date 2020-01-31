## Genetic Algorithm for salesman problem

### EN

The script uses TSPLIB xml data

https://www.iwr.uni-heidelberg.de/groups/comopt/software/TSPLIB95/XML-TSPLIB/instances/

params:

--input-file=`filename`

--roulette=[value] `default: False`

--init-size=12 

--selection-size=6 

--generations=500 

--mutation-rate=4

--test-repetitions=10

### Selection method:

It's possible to use two selection method:
- Roulette wheel (activable by `--roulette=True`, `--selection-size` will be ignored)
- Fixed selection (activable by `--selection-size` and `--roulette=False`)
  
### IT

Lo script estrae i dati dalla struttura xml di TSPLIB

https://www.iwr.uni-heidelberg.de/groups/comopt/software/TSPLIB95/XML-TSPLIB/instances/

paramentri:

--input-file=`filename` "nome del file XML contenete il grafo delle stazioni"

--roulette=[value] `default: False`

--init-size=12 "dimensione della prima generazione"

--generations=500 "generazioni" 

--selection-size=6 "rate di selezione"

--mutation-rate=4 "rate di mutazione"

--test-repetitions=10 "rate ripetizione del test"

### Metodo di selezione:

E' possibile utilizzare due metodi per la selezione dei geni:
- Roulette wheel (attivabile trmite il parametro `--roulette=True` il parametro `--selection-size` sarà ignorato)
    - La probabilità di riproduzione dell'elemento è data dal rapporto tra il suo valore di fitness e il totale dei valori di fitness. In base alle probabilità di riproduzione l'algoritmo seleziona casualmente due coppie A e B ( selezione casuale )
- Selezione di dimensione fissa (attivabile trmite il parametro `--selection-size` e `--roulette=False`)
    - viene selezionato un sottoinsieme della generazione dalla quale saranno generate coppie random
     
    