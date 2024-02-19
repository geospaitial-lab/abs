# Dokumentation der Konfigurationsdatei

Die Werte der folgenden Konfigurationsdatei entsprechen denen der [Beispielkonfiguration](../example_config.yaml).  
Klicken Sie auf die Felder, um die entsprechenden Erklärungen anzuzeigen.

<pre>
<a href="#eingangsdaten">EINGANGSDATEN</a>:
  <a href="#panoramas">PANORAMAS</a>:
    <a href="#verzeichnis-panoramas">VERZEICHNIS</a>: /Pfad/zum/Basisverzeichnis/der/Panoramas
    <a href="#hoehe-und-breite">HOEHE</a>:
    <a href="#hoehe-und-breite">BREITE</a>:
    <a href="#aufnahmepunkte">AUFNAHMEPUNKTE</a>:
      <a href="#pfad">PFAD</a>: /Pfad/zu/Aufnahmepunkte.gpkg
      <a href="#id_basis">ID_BASIS</a>:
      <a href="#felder">FELDER</a>:
        <a href="#id">ID</a>: image_id
        <a href="#aufnahmezeitpunkt">AUFNAHMEZEITPUNKT</a>: recordedAt
        <a href="#aufnahmerichtung">AUFNAHMERICHTUNG</a>: recorderDi
        <a href="#aufnahmepitch">AUFNAHMEPITCH</a>:
        <a href="#aufnahmeroll">AUFNAHMEROLL</a>:
        <a href="#aufnahmehoehe">AUFNAHMEHOEHE</a>: height
        <a href="#panorama_pfad">PANORAMA_PFAD</a>:
        <a href="#masken_pfad-felder">MASKEN_PFAD</a>:
    <a href="#ausrichtung">AUSRICHTUNG</a>:
    <a href="#masken_pfad-panoramas">MASKEN_PFAD</a>:
  <a href="#punktwolken">PUNKTWOLKEN</a>:
    <a href="#verzeichnis-punktwolken">VERZEICHNIS</a>: /Pfad/zum/Basisverzeichnis/der/Punktwolken
    <a href="#id_typ">ID_TYP</a>:
    <a href="#id_trennung">ID_TRENNUNG</a>:
    <a href="#kachelgroesse">KACHELGROESSE</a>: 50
  <a href="#aggregationsflaechen">AGGREGATIONSFLAECHEN</a>: /Pfad/zu/Aggregationsflaechen.gpkg
  <a href="#pseudo_aufnahmespuren">PSEUDO_AUFNAHMESPUREN</a>:

<a href="#cache_ignorieren">CACHE_IGNORIEREN</a>:

<a href="#ausgabeverzeichnis">AUSGABEVERZEICHNIS</a>: /Pfad/zum/Ausgabeverzeichnis

<a href="#orthos_speichern">ORTHOS_SPEICHERN</a>:

<a href="#epsg_code">EPSG_CODE</a>:

<a href="#nachverarbeitung">NACHVERARBEITUNG</a>:
  <a href="#geometrien_vereinfachen">GEOMETRIEN_VEREINFACHEN</a>:
  <a href="#strassenmarkierungen">STRASSENMARKIERUNGEN</a>:
    <a href="#schwellenwerte">SCHWELLENWERTE</a>:
      <a href="#einzelmarkierungen">EINZELMARKIERUNGEN</a>:
        <a href="#schlecht">SCHLECHT</a>: 0.25
        <a href="#mittel">MITTEL</a>: 0.25
  <a href="#strassenzustand">STRASSENZUSTAND</a>:
    <a href="#normierungsfaktoren">NORMIERUNGSFAKTOREN</a>:
      <a href="#asphalt-normierungsfaktoren">ASPHALT</a>:
        <a href="#sm4l_m-asphalt">SM4L_M</a>:
          FUNKTIONSKLASSE_A:
            1_5_Wert: 4.0
            WARNWERT: 12.0
            SCHWELLENWERT: 16.0
          FUNKTIONSKLASSE_B:
            1_5_Wert: 4.0
            WARNWERT: 16.0
            SCHWELLENWERT: 25.0
        <a href="#sm4l_a-asphalt">SM4L_A</a>:
          FUNKTIONSKLASSE_A:
            1_5_Wert: 1.0
            WARNWERT: 2.5
            SCHWELLENWERT: 3.5
          FUNKTIONSKLASSE_B:
            1_5_Wert: 1.0
            WARNWERT: 3.5
            SCHWELLENWERT: 5.0
        <a href="#mspt-asphalt">MSPT</a>:
          FUNKTIONSKLASSE_A:
            1_5_Wert: 4.0
            WARNWERT: 15.0
            SCHWELLENWERT: 25.0
          FUNKTIONSKLASSE_B:
            1_5_Wert: 4.0
            WARNWERT: 15.0
            SCHWELLENWERT: 25.0
        <a href="#msph-asphalt">MSPH</a>:
          FUNKTIONSKLASSE_A:
            1_5_Wert: 0.1
            WARNWERT: 4.0
            SCHWELLENWERT: 6.0
          FUNKTIONSKLASSE_B:
            1_5_Wert: 0.1
            WARNWERT: 4.0
            SCHWELLENWERT: 6.0
        <a href="#riss">RISS</a>:
          FUNKTIONSKLASSE_A:
            1_5_Wert: 1
            WARNWERT: 15
            SCHWELLENWERT: 25
          FUNKTIONSKLASSE_B:
            1_5_Wert: 5
            WARNWERT: 20
            SCHWELLENWERT: 33
          FUNKTIONSKLASSE_N:
            1_5_Wert: 5
            WARNWERT: 20
            SCHWELLENWERT: 33
        <a href="#afli">AFLI</a>:
          FUNKTIONSKLASSE_A:
            1_5_Wert: 1
            WARNWERT: 15
            SCHWELLENWERT: 25
          FUNKTIONSKLASSE_B:
            1_5_Wert: 5
            WARNWERT: 20
            SCHWELLENWERT: 33
          FUNKTIONSKLASSE_N:
            1_5_Wert: 5
            WARNWERT: 20
            SCHWELLENWERT: 33
        <a href="#ofs-asphalt">OFS</a>:
          FUNKTIONSKLASSE_A:
            1_5_Wert: 1
            WARNWERT: 15
            SCHWELLENWERT: 25
          FUNKTIONSKLASSE_B:
            1_5_Wert: 10
            WARNWERT: 33
            SCHWELLENWERT: 50
          FUNKTIONSKLASSE_N:
            1_5_Wert: 10
            WARNWERT: 33
            SCHWELLENWERT: 50
      <a href="#pflaster_platten-normierungsfaktoren">PFLASTER_PLATTEN</a>:
        <a href="#sm4l_m-pflaster_platten">SM4L_M</a>:
          FUNKTIONSKLASSE_A:
            1_5_Wert: 4.0
            WARNWERT: 12.0
            SCHWELLENWERT: 16.0
          FUNKTIONSKLASSE_B:
            1_5_Wert: 4.0
            WARNWERT: 16.0
            SCHWELLENWERT: 25.0
        <a href="#sm4l_a-pflaster_platten">SM4L_A</a>:
          FUNKTIONSKLASSE_A:
            1_5_Wert: 1.0
            WARNWERT: 2.5
            SCHWELLENWERT: 3.5
          FUNKTIONSKLASSE_B:
            1_5_Wert: 1.0
            WARNWERT: 3.5
            SCHWELLENWERT: 5.0
        <a href="#mspt-pflaster_platten">MSPT</a>:
          FUNKTIONSKLASSE_A:
            1_5_Wert: 4.0
            WARNWERT: 15.0
            SCHWELLENWERT: 25.0
          FUNKTIONSKLASSE_B:
            1_5_Wert: 4.0
            WARNWERT: 15.0
            SCHWELLENWERT: 25.0
        <a href="#msph-pflaster_platten">MSPH</a>:
          FUNKTIONSKLASSE_A:
            1_5_Wert: 0.1
            WARNWERT: 4.0
            SCHWELLENWERT: 6.0
          FUNKTIONSKLASSE_B:
            1_5_Wert: 0.1
            WARNWERT: 4.0
            SCHWELLENWERT: 6.0
        <a href="#ofs-pflaster_platten">OFS</a>:
          FUNKTIONSKLASSE_A:
            1_5_Wert: 1
            WARNWERT: 15
            SCHWELLENWERT: 25
          FUNKTIONSKLASSE_B:
            1_5_Wert: 5
            WARNWERT: 25
            SCHWELLENWERT: 40
          FUNKTIONSKLASSE_N:
            1_5_Wert: 5
            WARNWERT: 25
            SCHWELLENWERT: 40
    <a href="#gewichtungsfaktoren">GEWICHTUNGSFAKTOREN</a>:
      <a href="#asphalt-gewichtungsfaktoren">ASPHALT</a>:
        <a href="#gebrauchswert-asphalt">GEBRAUCHSWERT</a>:
          <a href="#zwsm4l-asphalt">ZWSM4L</a>: 0.25
          <a href="#zwmspt-asphalt">ZWMSPT</a>: 0.5
          <a href="#zwmsph-asphalt">ZWMSPH</a>: 0.25
        <a href="#substanzwert-asphalt">SUBSTANZWERT</a>:
          <a href="#zwriss">ZWRISS</a>: 0.56
          <a href="#zwafli">ZWAFLI</a>: 0.31
          <a href="#zwofs">ZWOFS</a>: 0.13
      <a href="#pflaster_platten-gewichtungsfaktoren">PFLASTER_PLATTEN</a>:
        <a href="#gebrauchswert-pflaster_platten">GEBRAUCHSWERT</a>:
          <a href="#zwsm4l-pflaster_platten">ZWSM4L</a>: 0.25
          <a href="#zwmspt-pflaster_platten">ZWMSPT</a>: 0.5
          <a href="#zwmsph-pflaster_platten">ZWMSPH</a>: 0.25
</pre>

## EINGANGSDATEN

Unter `EINGANGSDATEN` werden alle Eingangsdaten spezifiziert.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

### PANORAMAS

Unter `PANORAMAS` werden alle Informationen bezüglich der Panoramas spezifiziert.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

#### <a id="verzeichnis-panoramas"></a> VERZEICHNIS

Hier ist der Pfad zum Basisverzeichnis anzugeben, in dem sich alle Panoramas befinden.  
Die Panoramas müssen als `.jpg`-Dateien vorliegen, nach ihren IDs benannt sein und sich in einem
nach den ersten 6 Zeichen der IDs benannten Unterverzeichnis befinden (*Cyclomedia*).  
Das Panorama mit der ID `ABCD1234` muss sich beispielsweise unter `<VERZEICHNIS>/ABCD12/ABCD1234.jpg` befinden.  
Alternativ kann im Feld [`PANORAMA_PFAD`](#panorama_pfad) der [`AUFNAHMEPUNKTE`](#aufnahmepunkte) für jedes Panorama
ein relativer Pfad von dem hier angegebenen Basisverzeichnis spezifiziert werden.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

---

#### HOEHE und BREITE

Diese Felder sind **optional**. Standard: wird automatisch ermittelt  
Hier sind die Höhe und die Breite der Panoramas in Pixeln anzugeben.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

---

#### AUFNAHMEPUNKTE

Unter `AUFNAHMEPUNKTE` werden die Geodaten spezifiziert, welche die Aufnahmepunkte und weitere
Metadaten der Panoramas enthalten.
Jedes Panorama muss durch eine Punktgeometrie abgebildet sein, welche die im folgenden
aufgelisteten [`FELDER`](#felder) enthält.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

---

##### PFAD

Hier ist der Pfad zur `.shp`- oder `.gpkg`-Datei der Aufnahmepunkte anzugeben.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

---

##### <a id="id_basis"></a> ID_BASIS

Dieses Feld ist **optional**. Standard: 10  
Hier ist anzugeben, wie die IDs der Panoramas im Feld [`ID`](#id) codiert sind.  
*Cyclomedia* verwendet beispielsweise eine 8-stellige alphanumerische Basis-36 ID: `ABCD1234`.
Eine Basis-10 ID wäre hingegen `12345678`.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

---

##### FELDER

Unter `FELDER` werden die Felder der Geodaten spezifiziert.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

###### ID

Hier ist der Name des Felds anzugeben, welches die ID des Panoramas enthält.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

###### AUFNAHMEZEITPUNKT

Hier ist der Name des Felds anzugeben, welches den Aufnahmezeitpunkt des Panoramas enthält.  
Der Aufnahmezeitpunkt muss im ISO-8601 Format codiert sein: `2024-01-01T12:00:00.0000000+01:00`.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

###### AUFNAHMERICHTUNG

Hier ist der Name des Felds anzugeben, welches die Fahrtrichtung des Aufnahmefahrzeugs
zum Aufnahmezeitpunkt (Yaw) enthält.  
Die Aufnahmerichtung wird durch den relativen Winkel zu Norden in Grad angegeben.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

###### AUFNAHMEPITCH

Dieses Feld ist **optional**.  
Hier ist der Name des Felds anzugeben, welches den Pitch-Winkel des Aufnahmefahrzeugs
zum Aufnahmezeitpunkt enthält.
Der Pitch-Winkel wird in Grad angegeben.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

###### AUFNAHMEROLL

Dieses Feld ist **optional**.  
Hier ist der Name des Felds anzugeben, welches den Roll-Winkel des Aufnahmefahrzeugs
zum Aufnahmezeitpunkt enthält.
Der Roll-Winkel wird in Grad angegeben.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

###### AUFNAHMEHOEHE

Hier ist der Name des Felds anzugeben, welches die Aufnahmehöhe des Panoramas enthält.  
Es ist die absolute Höhe des Brennpunkts der Kamera im selben Höhensystem, welches
in den Punktwolken verwendet wird, anzugeben.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

###### <a id="panorama_pfad"></a> PANORAMA_PFAD

Dieses Feld ist **optional**.  
Hier ist der Name des Felds anzugeben, welches den relativen Pfad vom [`VERZEICHNIS`](#verzeichnis-panoramas)
der Panoramas zum Panorama enthält.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

###### <a id="masken_pfad-felder"></a> MASKEN_PFAD

Dieses Feld ist **optional**.  
Hier ist der Name des Felds anzugeben, welches den relativen Pfad vom [`VERZEICHNIS`](#verzeichnis-panoramas)
der Panoramas zur Maske enthält.  
Diese maskiert störende Objekte, wie das Aufnahmefahrzeug oder andere Verkehrsteilnehmende.  
Die Maske muss als `.png`-Datei mit 3 Kanälen oder als `.jpg`-Datei vorliegen.
Pixel, welche zu störenden Objekten gehören, werden mit dem Wert `[255, 255, 255]` maskiert.
Alle anderen Pixel haben den Wert `[0, 0, 0]`.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

---

#### AUSRICHTUNG

Dieses Feld ist **optional**. Standard: *Geographic*  
Hier ist anzugeben, wie die Panoramas ausgerichtet sind.
Es sind folgende Optionen möglich:

- *Geographic:* Die Panoramas sind mit ihrer Bildmitte in Richtung geografischem Norden ausgerichtet (Standard)
- *Grid:* Die Panoramas sind mit ihrer Bildmitte in Richtung Gitter-Norden ausgerichtet
- *Recording_Direction:* Die Panoramas sind mit ihrer Bildmitte in Fahrtrichtung des Aufnahmefahrzeugs ausgerichtet
  (relativ zu geografischem Norden)
- *Recording_Direction_Grid:* Die Panoramas sind mit ihrer Bildmitte in Fahrtrichtung des Aufnahmefahrzeugs ausgerichtet
  (relativ zu Gitter-Norden)

[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

---

#### <a id="masken_pfad-panoramas"></a> MASKEN_PFAD

Dieses Feld ist **optional**. Standard: [*data/images/car_mask.png*](../data/images/car_mask.png)  
Hier ist der Pfad zur Maske anzugeben, welche das Aufnahmefahrzeug maskiert.  
Die Maske muss als `.png`-Datei mit 3 Kanälen oder als `.jpg`-Datei vorliegen.
Pixel, welche zum Aufnahmefahrzeug gehören, werden mit dem Wert `[255, 255, 255]` maskiert.
Alle anderen Pixel haben den Wert `[0, 0, 0]`.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

---

### PUNKTWOLKEN

Unter `PUNKTWOLKEN` werden alle Informationen bezüglich der Punktwolken spezifiziert.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

#### <a id="verzeichnis-punktwolken"></a> VERZEICHNIS

Hier ist der Pfad zum Basisverzeichnis anzugeben, in dem sich alle Punktwolken befinden.  
Die Punktwolken müssen als `.las`- oder `.laz`-Dateien vorliegen und dürfen sich nicht in Unterverzeichnissen befinden.
Die Dateinamen der Punktwolken müssen eine ID enthalten, welche x- und y-Koordinate der südwestlichen
Ecke des von der Punktwolke abgedeckten Bereichs codiert.  
Der Dateiname muss die Form `<beliebig><x-Koordinate>ID_TRENNUNG<y-Koordinate><beliegbig>.las/.laz` haben:
`punktwolke_1234_123456_abc.laz`  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

---

#### <a id="id_typ"></a> ID_TYP

Dieses Feld ist **optional**. Standard: *Numbered*  
Hier ist anzugeben, in welcher Form x- und y-Koordinate in der ID im Dateinamen der Punktwolken enthalten sind.  
Es sind folgende Optionen möglich:

- *Numbered:* Die Koordinaten sind durch die [`KACHELGROESSE`](#kachelgroesse) geteilt enthalten
(Standard, *Cyclomedia*)
- *Coordinates:* Die Koordinaten sind direkt enthalten

[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

---

#### <a id="id_trennung"></a> ID_TRENNUNG

Dieses Feld ist **optional**. Standard: *_*  
Hier ist anzugeben, durch welche(s) Trennzeichen die x- und y-Koordinate in der ID getrennt sind.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

---

#### KACHELGROESSE

Hier ist anzugeben, wie groß eine Seite des quadratischen, von einer Punktwolke abgedeckten, Bereichs ist.  
Es können nur ganzzahlige Werte in Metern angegeben werden.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

---

### AGGREGATIONSFLAECHEN

Auf die Aggregationsflächen werden die berechneten Substanz- und Ebenheitswerte gemäß *AP9 K* aggregiert und bewertet.  
Hier ist der Pfad zur `.shp`- oder `.gpkg`-Datei der Aggregationsflächen anzugeben.  
Die Geodaten dürfen ausschließlich Polygon-Geometrien enthalten, wobei jede Geometrie
die Felder *FK* und *BW* enthalten muss.  
Das Feld *FK* (Funktionsklasse) muss einen der folgenden Werte enthalten:
- *A:* Hauptverkehrsstraße
- *B:* Nebenstraße
- *N:* Nebenfläche

Das Feld *BW* (Bauweise) muss einen der folgenden Werte enthalten:
- *A:* Asphalt
- *P:* Pflaster/ Platten

Die Aggregationsflächen dürfen zudem beliebige weitere Felder enthalten.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

### <a id="pseudo_aufnahmespuren"></a> PSEUDO_AUFNAHMESPUREN

Dieses Feld ist **optional**.  
Die Pseudo-Aufnahmespuren dienen zur zusätzlichen Berechnung von Ebenheitswerten.
Diese werden andernfalls ausschließlich für die aus den Aufnahmepunkten abgeleiteten Fahrspuren berechnet.  
Hier ist der Pfad zur `.shp`- oder `.gpkg`-Datei der Pseudo-Aufnahmespuren anzugeben.
Die Geodaten dürfen ausschließlich Linien-Geometrien enthalten.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

## <a id="cache_ignorieren"></a> CACHE_IGNORIEREN

Dieses Feld ist **optional**. Standard: *False*  
Es sind folgende Optionen möglich:

- *True:* bereits verarbeitete Bereiche werden neu berechnet
- *False:* bereits verarbeitete Bereiche werden nicht neu berechnet (Standard)

[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

## AUSGABEVERZEICHNIS

Hier ist der Pfad zum Ausgabeverzeichnis anzugeben.  
Es kann ein bereits verwendetes Ausgabeverzeichnis zum Fortsetzen der Berechnungen angegeben werden.  
Wenn das Verzeichnis nicht existiert, wird es erstellt.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

## <a id="orthos_speichern"></a> ORTHOS_SPEICHERN

Dieses Feld ist **optional**. Standard: *True*  
Es sind folgende Optionen möglich:

- *True:* Die bei der Berechnung erstellten Orthos werden gespeichert (Standard)
- *False:* Die bei der Berechnung erstellten Orthos werden nicht gespeichert

[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

## <a id="epsg_code"></a> EPSG_CODE

Dieses Feld ist **optional**. Standard: *25832*  
Hier ist der EPSG-Code des Koordinatenreferenzsystems anzugeben, in dem sich alle Ausgabedaten befinden sollen.  
Der EPSG-Code muss mit dem der Eingangsdaten übereinstimmen.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

## NACHVERARBEITUNG

Diese Felder sind **optional**.  
Unter `NACHVERARBEITUNG` werden die Schwellenwerte und Normierungs- und Gewichtungsfaktoren für die Bewertung
spezifiziert.  
**Alle** Felder sind **optional** und haben standardmäßig die Werte aus der [Beispielkonfiguration](../example_config.yaml).  
Die Werte unter [`STRASSENZUSTAND`](#strassenzustand) entsprechen der *AP9 K*.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

### <a id="geometrien_vereinfachen"></a> GEOMETRIEN_VEREINFACHEN

Dieses Feld ist **optional**. Standard: *False*  
Es sind folgende Optionen möglich:

- *True:* Die ausgegebenen Geometrien der Geodaten (`Strassenmarkierungen.gpkg`, `Substanz.gpkg`) werden mit dem
Douglas-Peucker-Algorithmus vereinfacht (Reduzierung der Datenmenge)
- *False:* Die ausgegebenen Geometrien der Geodaten (`Strassenmarkierungen.gpkg`, `Substanz.gpkg`) werden
nicht vereinfacht (Standard)

[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

### STRASSENMARKIERUNGEN

Unter `STRASSENMARKIERUNGEN` werden die Parameter für die Bewertung der Straßenmarkierungen spezifiziert.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

#### SCHWELLENWERTE

Unter `SCHWELLENWERTE` werden die Schwellenwerte für die Bewertung der Straßenmarkierungen spezifiziert.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

##### EINZELMARKIERUNGEN

Unter `EINZELMARKIERUNGEN` werden die Schwellenwerte für die Bewertung der Einzelmarkierungen spezifiziert.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

###### SCHLECHT

Standard: *0.25*  
Hier ist anzugeben, ab welchem Anteil der Markierungsfläche, die den Zustand *schlecht* hat,
eine Markierung in jedem Fall als *schlecht* bewertet wird.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

###### MITTEL

Standard: *0.25*  
Hier ist anzugeben, ab welchem Anteil der Markierungsfläche, die den Zustand *schlecht* oder *mittel* hat,
eine Markierung als *mittel* bewertet wird.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

---

### STRASSENZUSTAND

Unter `STRASSENZUSTAND` werden die Parameter für die Bewertung des Straßenzustands spezifiziert.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

#### NORMIERUNGSFAKTOREN

Unter `NORMIERUNGSFAKTOREN` werden die Normierungsfaktoren für die Bewertung des Straßenzustands spezifiziert.  
Durch die Normierung werden die dimensionsbehafteten Zustandsgrößen in dimensionslose Zustandswerte
für jede Aggregationsfläche überführt.  
Ein Zustandswert ist ein Wert zwischen 1.0 und 5.0, wobei 1.0 den besten und 5.0 den schlechtesten Zustand darstellt.  
Jede Zustandsgröße hat in Abhängigkeit der Bauweise und der Funktionsklasse 3 Normierungsfaktoren
`1_5_Wert`, `WARNWERT` und `SCHWELLENWERT`, mit denen sich die Sensitivität der Bewertung einstellen lässt.  
Gegebenenfalls müssen Sie die Normierungsfaktoren an Ihre Anforderungen anpassen.  
Weitere Informationen finden Sie in der *AP9 K*.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

##### <a id="asphalt-normierungsfaktoren"></a> ASPHALT

Unter `ASPHALT` werden die Normierungsfaktoren für die Bewertung des Straßenzustands von Straßen
der Bauweise Asphalt spezifiziert.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

###### <a id="sm4l_m-asphalt"></a> SM4L_M

Unter `SM4L_M` werden die Normierungsfaktoren für die Zustandsgröße *SM4L_M* (Ebenheitswert) für verschiedene
Funktionsklassen spezifiziert.  
Diese entspricht dem Maximum der Stichmaße in Lattenmitte unter der 4m-Latte in mm.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

###### <a id="sm4l_a-asphalt"></a> SM4L_A

Unter `SM4L_A` werden die Normierungsfaktoren für die Zustandsgröße *SM4L_A* (Ebenheitswert) für verschiedene
Funktionsklassen spezifiziert.  
Diese entspricht dem Mittelwert der Stichmaße in Lattenmitte unter der 4m-Latte in mm.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

###### <a id="mspt-asphalt"></a> MSPT

Unter `MSPT` werden die Normierungsfaktoren für die Zustandsgröße *MSPT* (Ebenheitswert) für verschiedene
Funktionsklassen spezifiziert.  
Diese entspricht dem Maximum der Mittelwerte der linken und rechten Spurrinnentiefen in mm.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

###### <a id="msph-asphalt"></a> MSPH

Unter `MSPH` werden die Normierungsfaktoren für die Zustandsgröße *MSPH* (Ebenheitswert) für verschiedene
Funktionsklassen spezifiziert.  
Diese entspricht dem Maximum der Mittelwerte der linken und rechten fiktiven Wassertiefen in mm.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

###### <a id="riss"></a> RISS

Unter `RISS` werden die Normierungsfaktoren für die Zustandsgröße *RISS* (Substanzwert) für verschiedene
Funktionsklassen spezifiziert.  
Diese entspricht dem Anteil der durch Risse betroffenen Fläche in %.  
**HINWEIS:** Die Schadensobjekte werden auf ein 1m x 1m Raster abgebildet.
Jedes von einem Schadensobjekt betroffenes Feld entspricht einer schadhaften Fläche.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

###### <a id="afli"></a> AFLI

Unter `AFLI` werden die Normierungsfaktoren für die Zustandsgröße *AFLI* (Substanzwert) für verschiedene
Funktionsklassen spezifiziert.  
Diese entspricht dem Anteil der durch aufgelegte Flickstellen betroffenen Fläche in %.  
**HINWEIS:** Die Schadensobjekte werden auf ein 1m x 1m Raster abgebildet.
Jedes von einem Schadensobjekt betroffenes Feld entspricht einer schadhaften Fläche.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

###### <a id="ofs-asphalt"></a> OFS

Unter `OFS` werden die Normierungsfaktoren für die Zustandsgröße *OFS* (Substanzwert) für verschiedene
Funktionsklassen spezifiziert.  
Diese entspricht dem Anteil der durch sonstige Oberflächenschäden (Abplatzungen/ Ausbrüche,
Offene Nähte/ Fugen) betroffenen Fläche in %.  
**HINWEIS:** Die Schadensobjekte werden auf ein 1m x 1m Raster abgebildet.
Jedes von einem Schadensobjekt betroffenes Feld entspricht einer schadhaften Fläche.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

##### <a id="pflaster_platten-normierungsfaktoren"></a> PFLASTER_PLATTEN

Unter `PFLASTER_PLATTEN` werden die Normierungsfaktoren für die Bewertung des Straßenzustands von Straßen
der Bauweise Pflaster/ Platten spezifiziert.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

###### <a id="sm4l_m-pflaster_platten"></a> SM4L_M

Unter `SM4L_M` werden die Normierungsfaktoren für die Zustandsgröße *SM4L_M* (Ebenheitswert) für verschiedene
Funktionsklassen spezifiziert.  
Diese entspricht dem Maximum der Stichmaße in Lattenmitte unter der 4m-Latte in mm.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

###### <a id="sm4l_a-pflaster_platten"></a> SM4L_A

Unter `SM4L_A` werden die Normierungsfaktoren für die Zustandsgröße *SM4L_A* (Ebenheitswert) für verschiedene
Funktionsklassen spezifiziert.  
Diese entspricht dem Mittelwert der Stichmaße in Lattenmitte unter der 4m-Latte in mm.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

###### <a id="mspt-pflaster_platten"></a> MSPT

Unter `MSPT` werden die Normierungsfaktoren für die Zustandsgröße *MSPT* (Ebenheitswert) für verschiedene
Funktionsklassen spezifiziert.  
Diese entspricht dem Maximum der Mittelwerte der linken und rechten Spurrinnentiefen in mm.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

###### <a id="msph-pflaster_platten"></a> MSPH

Unter `MSPH` werden die Normierungsfaktoren für die Zustandsgröße *MSPH* (Ebenheitswert) für verschiedene
Funktionsklassen spezifiziert.  
Diese entspricht dem Maximum der Mittelwerte der linken und rechten fiktiven Wassertiefen in mm.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

###### <a id="ofs-pflaster_platten"></a> OFS

Unter `OFS` werden die Normierungsfaktoren für die Zustandsgröße *OFS* (Substanzwert) für verschiedene
Funktionsklassen spezifiziert.  
Diese entspricht dem Anteil der durch sonstige Oberflächenschäden (**HINWEIS:** momentan werden
keine Substanzwerte für die Bauweise Pflaster/ Platten erfasst!) betroffenen Fläche in %.  
**HINWEIS:** Die Schadensobjekte werden auf ein 1m x 1m Raster abgebildet.
Jedes von einem Schadensobjekt betroffenes Feld entspricht einer schadhaften Fläche.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

---

#### GEWICHTUNGSFAKTOREN

Unter `GEWICHTUNGSFAKTOREN` werden die Gewichtungsfaktoren für die Bewertung des Straßenzustands spezifiziert.  
Durch die Gewichtung werden die dimensionslosen Zustandswerte für jede Aggregationsfläche zu Teilzielwerten und
einem Gesamtwert verknüpft.  
Ein Teilziel- bzw. Gesamtwert ist ein Wert zwischen 1.0 und 5.0, wobei 1.0 den besten und 5.0 den
schlechtesten Zustand darstellt.  
Jeder Zustandswert hat einen Gewichtungsfaktor, mit dem sich sein Einfluss auf die Bewertung einstellen lässt.  
Gegebenenfalls müssen Sie die Gewichtungsfaktoren an Ihre Anforderungen anpassen.  
Weitere Informationen finden Sie in der *AP9 K*.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

##### <a id="asphalt-gewichtungsfaktoren"></a> ASPHALT

Unter `ASPHALT` werden die Gewichtungsfaktoren für die Bewertung des Straßenzustands von Straßen
der Bauweise Asphalt spezifiziert.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

###### <a id="gebrauchswert-asphalt"></a> GEBRAUCHSWERT

Unter `GEBRAUCHSWERT` werden die Gewichtungsfaktoren für den Teilzielwert *TWGEB* spezifiziert.  
Dieser verknüpft die Zustandswerte der Ebenheitswerte.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

###### <a id="zwsm4l-asphalt"></a> ZWSM4L

Unter `ZWSM4L` wird der Gewichtungsfaktor für die Zustandsgröße *ZWSM4L* spezifiziert.  
Diese entspricht dem Maximum von *ZWSM4L_M* (Zustandswert (Maximum der Stichmaße in Lattenmitte unter der 4m-Latte))
und *ZWSM4L_A* (Zustandswert (Mittelwert der Stichmaße in Lattenmitte unter der 4m-Latte)).  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

---

###### <a id="zwmspt-asphalt"></a> ZWMSPT

Unter `ZWMSPT` wird der Gewichtungsfaktor für die Zustandsgröße *ZWMSPT* spezifiziert.  
Diese entspricht dem Zustandswert (Maximum der Mittelwerte der linken und rechten Spurrinnentiefen).  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

---

###### <a id="zwmsph-asphalt"></a> ZWMSPH

Unter `ZWMSPH` wird der Gewichtungsfaktor für die Zustandsgröße *ZWMSPH* spezifiziert.  
Diese entspricht dem Zustandswert (Maximum der Mittelwerte der linken und rechten fiktiven Wassertiefen).  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

---

###### <a id="substanzwert-asphalt"></a> SUBSTANZWERT

Unter `SUBSTANZWERT` werden die Gewichtungsfaktoren für den Teilzielwert *TWSUB* spezifiziert.  
Dieser verknüpft die Zustandswerte der Substanzwerte.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

###### <a id="zwriss"></a> ZWRISS

Unter `ZWRISS` wird der Gewichtungsfaktor für die Zustandsgröße *ZWRISS* spezifiziert.  
Diese entspricht dem Zustandswert (Anteil der durch Risse betroffenen Fläche).  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

---

###### <a id="zwafli"></a> ZWAFLI

Unter `ZWAFLI` wird der Gewichtungsfaktor für die Zustandsgröße *ZWAFLI* spezifiziert.  
Diese entspricht dem Zustandswert (Anteil der durch aufgelegte Flickstellen betroffenen Fläche).  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

---

###### <a id="zwofs"></a> ZWOFS

Unter `ZWOFS` wird der Gewichtungsfaktor für die Zustandsgröße *ZWOFS* spezifiziert.  
Diese entspricht dem Zustandswert (Anteil der durch sonstige Oberflächenschäden (Abplatzungen/ Ausbrüche,
Offene Nähte/ Fugen) betroffenen Fläche).  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

---

##### <a id="pflaster_platten-gewichtungsfaktoren"></a> PFLASTER_PLATTEN

Unter `PFLASTER_PLATTEN` werden die Gewichtungsfaktoren für die Bewertung des Straßenzustands von Straßen
der Bauweise Pflaster/ Platten spezifiziert.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

###### <a id="gebrauchswert-pflaster_platten"></a> GEBRAUCHSWERT

Unter `GEBRAUCHSWERT` werden die Gewichtungsfaktoren für den Teilzielwert *TWGEB* spezifiziert.  
Dieser verknüpft die Zustandswerte der Ebenheitswerte.  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

###### <a id="zwsm4l-pflaster_platten"></a> ZWSM4L

Unter `ZWSM4L` wird der Gewichtungsfaktor für die Zustandsgröße *ZWSM4L* spezifiziert.  
Diese entspricht dem Maximum von *ZWSM4L_M* (Zustandswert (Maximum der Stichmaße in Lattenmitte unter der 4m-Latte))
und *ZWSM4L_A* (Zustandswert (Mittelwert der Stichmaße in Lattenmitte unter der 4m-Latte)).  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

---

###### <a id="zwmspt-pflaster_platten"></a> ZWMSPT

Unter `ZWMSPT` wird der Gewichtungsfaktor für die Zustandsgröße *ZWMSPT* spezifiziert.  
Diese entspricht dem Zustandswert (Maximum der Mittelwerte der linken und rechten Spurrinnentiefen).  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

---

###### <a id="zwmsph-pflaster_platten"></a> ZWMSPH

Unter `ZWMSPH` wird der Gewichtungsfaktor für die Zustandsgröße *ZWMSPH* spezifiziert.  
Diese entspricht dem Zustandswert (Maximum der Mittelwerte der linken und rechten fiktiven Wassertiefen).  
[↑ Zurück zur Übersicht](#dokumentation-der-konfigurationsdatei)

---