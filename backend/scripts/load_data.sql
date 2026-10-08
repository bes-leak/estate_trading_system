-- загрузка выгрузки из src/train_model/data/sql, папка примонтирована в контейнер как /data
-- запуск: docker compose exec db psql -U kvarts -d estate -f /backend/scripts/load_data.sql

BEGIN;

TRUNCATE listings, buildings RESTART IDENTITY CASCADE;

CREATE TEMP TABLE staging_buildings (
    geo_lat DOUBLE PRECISION,
    geo_lon DOUBLE PRECISION,
    build_year REAL,
    distance_metro REAL,
    distance_rail REAL,
    distance_major_road REAL,
    road_load_300 REAL,
    road_load_1000 REAL,
    dist_to_center REAL,
    distance_industrial REAL,
    industrial_area_1000 REAL,
    distance_park REAL,
    green_area_500 REAL,
    green_area_1000 REAL,
    distance_water REAL,
    ndvi_mean_300 REAL,
    ndvi_mean_500 REAL,
    ndvi_mean_1000 REAL,
    ndbi_mean_300 REAL,
    ndbi_mean_500 REAL,
    ndbi_mean_1000 REAL
) ON COMMIT DROP;

COPY staging_buildings FROM '/data/buildings.csv' CSV HEADER;

INSERT INTO buildings (
    geo_lat, geo_lon, geom, build_year,
    distance_metro, distance_rail, distance_major_road,
    road_load_300, road_load_1000, dist_to_center,
    distance_industrial, industrial_area_1000,
    distance_park, green_area_500, green_area_1000, distance_water,
    ndvi_mean_300, ndvi_mean_500, ndvi_mean_1000,
    ndbi_mean_300, ndbi_mean_500, ndbi_mean_1000
)
SELECT
    geo_lat, geo_lon,
    -- внимание на порядок: ST_MakePoint(долгота, широта)
    ST_SetSRID(ST_MakePoint(geo_lon, geo_lat), 4326),
    build_year::SMALLINT,
    distance_metro, distance_rail, distance_major_road,
    road_load_300, road_load_1000, dist_to_center,
    distance_industrial, industrial_area_1000,
    distance_park, green_area_500, green_area_1000, distance_water,
    ndvi_mean_300, ndvi_mean_500, ndvi_mean_1000,
    ndbi_mean_300, ndbi_mean_500, ndbi_mean_1000
FROM staging_buildings;

CREATE TEMP TABLE staging_listings (
    geo_lat DOUBLE PRECISION,
    geo_lon DOUBLE PRECISION,
    price BIGINT,
    area REAL,
    kitchen_area REAL,
    rooms SMALLINT,
    level SMALLINT,
    levels SMALLINT,
    building_type SMALLINT,
    object_type SMALLINT,
    published_at TIMESTAMP
) ON COMMIT DROP;

COPY staging_listings FROM '/data/listings.csv' CSV HEADER;

INSERT INTO listings (
    building_id, price, area, kitchen_area, rooms,
    level, levels, building_type, object_type, published_at
)
SELECT
    b.id, s.price, s.area, s.kitchen_area, s.rooms,
    s.level, s.levels, s.building_type, s.object_type, s.published_at
FROM staging_listings s
JOIN buildings b ON b.geo_lat = s.geo_lat AND b.geo_lon = s.geo_lon;

COMMIT;

ANALYZE buildings;
ANALYZE listings;

SELECT
    (SELECT count(*) FROM buildings) AS buildings,
    (SELECT count(*) FROM listings)  AS listings;
