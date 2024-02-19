# Dokumentation des Adapters *Geotechnik*

Führen Sie zunächst die beiden Skripte [`erzeuge_aufnahmepunkte.py`](erzeuge_aufnahmepunkte.py)
und [`punktwolken_kacheln.py`](punktwolken_kacheln.py) aus.

Das Skript [`erzeuge_aufnahmepunkte.py`](erzeuge_aufnahmepunkte.py) erzeugt aus den Metadaten
(`meta.json`-Dateien in Unterverzeichnissen) die Aufnahmepunkte.  
Das Metadaten-Verzeichnis ist mit dem Suffix `Rohdaten (Panoramabildstandorte)` benannt.

```
docker run -v /Pfad/zum/Metadaten-Verzeichnis:/meta_path -v /Pfad/zum/Ausgabeverzeichnis:/out_path abs python /abs/adapter/geotechnik/erzeuge_aufnahmepunkte.py /meta_path /out_path
```

Das Skript [`punktwolken_kacheln.py`](punktwolken_kacheln.py) teilt die Punktwolken neu ein und speichert diese.  
Das Metadaten-Verzeichnis ist mit dem Suffix `Rohdaten (Panoramabildstandorte)` benannt.  
Das Punktwolken-Verzeichnis ist mit dem Suffix `Rohdaten (Punktwolke)` benannt.  
Die Aggregationsflächen entsprechen denen, welche in der [`README.md`](../README.md) beschrieben sind.

```
docker run -v /Pfad/zum/Metadaten-Verzeichnis:/meta_path -v /Pfad/zum/Punktwolken-Verzeichnis:/las_path -v /Pfad/zu/Aggregationsflaechen.gpkg:/query_path.gpkg -v /Pfad/zum/Ausgabeverzeichnis:/out_path abs python /abs/adapter/geotechnik/punktwolken_kacheln.py /meta_path /las_path /query_path.gpkg /out_path
```

Im Anschluss können Sie dem Vorgehen in der [`README.md`](../README.md) folgen.  
Nutzen Sie als Beispiel-Konfigurationsdatei die [`example_config_geotechnik.yaml`](example_config_geotechnik.yaml).