# Dokumentation der Geodaten

## Straßenzustand

Die Geodaten `Straßenzustand.gpkg` enthalten die bewerteten Aggregationsflächen.  
**HINWEIS:** Die Felder mit dem Präfix *ZK*, *TK* und das Feld *GK* werden durch Klassenbildung aus den
Zustands-, Teilziel- und Gesamtwerten abgeleitet.
Eine Klasse ist ein Wert zwischen 1 und 4, wobei 1 den besten Zustand und 4 den schlechtesten Zustand darstellt.  
Weitere Informationen finden Sie in der *AP9 K*.

Die Geometrien haben folgende Felder:
- ursprüngliche Felder
- *SM4L_M:* Maximum der Stichmaße in Lattenmitte unter der 4m-Latte in mm
- *SM4L_A:* Mittelwert der Stichmaße in Lattenmitte unter der 4m-Latte in mm
- *S01:* Standardabweichung der Differenzen zum gleitenden Mittelwert bei einer Mittelungslänge von 1m in mm
- *S03:* Standardabweichung der Differenzen zum gleitenden Mittelwert bei einer Mittelungslänge von 3m in mm
- *S10:* Standardabweichung der Differenzen zum gleitenden Mittelwert bei einer Mittelungslänge von 10m in mm
- *S30:* Standardabweichung der Differenzen zum gleitenden Mittelwert bei einer Mittelungslänge von 30m in mm
- *LN:* Mittelwert der Längsneigungen in %
- *MSPTL:* Mittelwert der linken Spurrinnentiefen nach dem 1,2m-Latten-Prinzip in mm
- *MSPTR:* Mittelwert der rechten Spurrinnentiefen nach dem 1,2m-Latten-Prinzip in mm
- *MSPT:* Maximum der Mittelwerte der linken und rechten Spurrinnentiefen in mm
- *SPTMAX:* Maximum der Spurrinnentiefen in mm
- *SSPTL:* Standardabweichung der linken Spurrinnentiefen in mm
- *SSPTR:* Standardabweichung der rechten Spurrinnentiefen in mm
- *MSPHL:* Mittelwert der linken fiktiven Wassertiefen in mm
- *MSPHR:* Mittelwert der rechten fiktiven Wassertiefen in mm
- *MSPH:* Maximum der Mittelwerte der linken und rechten fiktiven Wassertiefen in mm
- *SSPHL:* Standardabweichung der linken fiktiven Wassertiefen in mm
- *SSPHR:* Standardabweichung der rechten fiktiven Wassertiefen in mm
- *QN:* Mittelwert der Querneigungen in %
- *RISS:* Anteil der durch Risse betroffenen Fläche in %
- *AFLI:* Anteil der durch aufgelegte Flickstellen betroffenen Fläche in %
- *OFS:* Anteil der durch sonstige Oberflächenschäden betroffenen Fläche in %
- *ZWSM4L_M:* Zustandswert (Maximum der Stichmaße in Lattenmitte unter der 4m-Latte)
- *ZWSM4L_A:* Zustandswert (Mittelwert der Stichmaße in Lattenmitte unter der 4m-Latte)
- *ZWSM4L:* Zustandswert (Maximum von ZWSM4L_M und ZWSM4L_A)
- *ZWMSPT:* Zustandswert (Maximum der Mittelwerte der linken und rechten Spurrinnentiefen)
- *ZWMSPH:* Zustandswert (Maximum der Mittelwerte der linken und rechten fiktiven Wassertiefen)
- *ZWRISS:* Zustandswert (Anteil der durch Risse betroffenen Fläche)
- *ZWAFLI:* Zustandswert (Anteil der durch aufgelegte Flickstellen betroffenen Fläche)
- *ZWOFS:* Zustandswert (Anteil der durch sonstige Oberflächenschäden betroffenen Fläche)
- *TWGEB:* Gebrauchswert
- *TWSUB:* Substanzwert
- *GW:* Gesamtwert (Maximum von TWGEB und TWSUB)
- *ZKSM4L_M:* Zustandsklasse (Maximum der Stichmaße in Lattenmitte unter der 4m-Latte)
- *ZKSM4L_A:* Zustandsklasse (Mittelwert der Stichmaße in Lattenmitte unter der 4m-Latte)
- *ZKSM4L:* Zustandsklasse (Maximum von ZWSM4L_M und ZWSM4L_A)
- *ZKMSPT:* Zustandsklasse (Maximum der Mittelwerte der linken und rechten Spurrinnentiefen)
- *ZKMSPH:* Zustandsklasse (Maximum der Mittelwerte der linken und rechten fiktiven Wassertiefen)
- *ZKRISS:* Zustandsklasse (Anteil der durch Risse betroffenen Fläche)
- *ZKAFLI:* Zustandsklasse (Anteil der durch aufgelegte Flickstellen betroffenen Fläche)
- *ZKOFS:* Zustandsklasse (Anteil der durch sonstige Oberflächenschäden betroffenen Fläche)
- *TKGEB:* Gebrauchsklasse
- *TKSUB:* Substanzklasse
- *GK:* Gesamtklasse (Maximum von TWGEB und TWSUB)

## Substanz

Die Geodaten `Substanz.gpkg` enthalten die Schadensobjekte als einzelne Polygon-Geometrien.

Die Geometrien haben folgende Felder:
- *typ:* Typ des Schadensobjekts
  - *r:* (Netz-)Riss
  - *aa:* Abplatzung/ Ausbruch
  - *af:* Aufgelegte Flickstelle
  - *onf:* Offene Naht/ Fuge
- *ebene:* Höhenebene des Objekts

## Straßenmarkierungen

Die Geodaten `Straßenmarkierungen.gpkg` enthalten die Straßenmarkierungen als einzelne Polygon-Geometrien.

Die Geometrien haben folgende Felder:
- *typ:* Typ der Straßenmarkierung
  - *Linie dünn*
  - *Linie dünn gestrichelt*
  - *Linie breit*
  - *Linie breit gestrichelt*
  - *Linie extrabreit*
  - *Pfeile*
  - *Symbole*
  - *Sperrflächen*
- *zustand:* Klassifizierter Gesamtzustand der Straßenmarkierung
  - *gut*
  - *mittel*
  - *schlecht*
- *gut:* Flächenmäßiger Anteil der Markierung, welcher als gut klassifiziert wurde in %
- *mittel:* Flächenmäßiger Anteil der Markierung, welcher als mittel klassifiziert wurde in %
- *schlecht:* Flächenmäßiger Anteil der Markierung, welcher als schlecht klassifiziert wurde in %
- *ebene:* Höhenebene des Objekts