CREATE EXTENSION IF NOT EXISTS postgis;

-- дом: одна строка на уникальную координату, средовые признаки общие для всех квартир в доме
CREATE TABLE IF NOT EXISTS buildings (
    id          BIGSERIAL PRIMARY KEY,
    geo_lat     DOUBLE PRECISION NOT NULL,
    geo_lon     DOUBLE PRECISION NOT NULL,
    geom        geometry(Point, 4326) NOT NULL,
    build_year  SMALLINT,

    distance_metro       REAL,
    distance_rail        REAL,
    distance_major_road  REAL,
    road_load_300        REAL,
    road_load_1000       REAL,
    dist_to_center       REAL,

    distance_industrial  REAL,
    industrial_area_1000 REAL,
    distance_park        REAL,
    green_area_500       REAL,
    green_area_1000      REAL,
    distance_water       REAL,

    ndvi_mean_300        REAL,
    ndvi_mean_500        REAL,
    ndvi_mean_1000       REAL,
    ndbi_mean_300        REAL,
    ndbi_mean_500        REAL,
    ndbi_mean_1000       REAL,

    UNIQUE (geo_lat, geo_lon)
);

CREATE TABLE IF NOT EXISTS listings (
    id            BIGSERIAL PRIMARY KEY,
    building_id   BIGINT NOT NULL REFERENCES buildings (id),
    price         BIGINT NOT NULL,
    area          REAL NOT NULL,
    kitchen_area  REAL,
    rooms         SMALLINT,
    level         SMALLINT,
    levels        SMALLINT,
    building_type SMALLINT,
    object_type   SMALLINT,
    published_at  TIMESTAMP NOT NULL
);

-- пространственный индекс: без него выборка по квадрату идёт полным перебором
CREATE INDEX IF NOT EXISTS buildings_geom_idx ON buildings USING GIST (geom);
CREATE INDEX IF NOT EXISTS listings_building_idx ON listings (building_id);
CREATE INDEX IF NOT EXISTS listings_price_idx ON listings (price);
