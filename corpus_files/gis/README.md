# AP district boundaries

Source: OpenStreetMap, pulled via the Nominatim API (nominatim.openstreetmap.org),
one relation per district, October 2026. Free, no login required, licensed
under ODbL (Open Database License, openstreetmap.org/copyright).

This is the only free source found that has all 28 current AP districts,
including Markapuram and Polavaram, alongside the 26 officially gazetted
districts. Other checked sources (data.gov.in, GADM, datta07/INDIAN-SHAPEFILES)
either had outdated 13-district boundaries or were missing the two newest
districts.

Files:
- `ap_districts_osm_shapefile/AP_DISTRICTS_OSM.shp` (+ .dbf/.shx/.prj/.cpg) —
  full-detail shapefile, EPSG:4326 (WGS84 lat/lon), one polygon per district.
- `ap_districts_osm.geojson` — same data as GeoJSON, full detail (not simplified).

The web map at `landing/assets/ap_districts.geojson` is a simplified,
coordinate-rounded copy of this data for faster page loads. Regenerate it
from these full-detail files if the map ever needs re-exporting.

Each feature has `district` (display name) and `data_key` (folder name
matching `corpus_files/district_data/<data_key>/`).
