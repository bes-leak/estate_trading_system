package postgres

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/bes-leak/estate_trading_system/backend/internal/domain"
)

const searchRadiusMeters = 25

// PostgreSQL error code for unique constraint violation
const uniqueViolation = "23505"

type BuildingRepo struct {
	pool *pgxpool.Pool
}

func NewBuildingRepo(pool *pgxpool.Pool) *BuildingRepo {
	return &BuildingRepo{pool: pool}
}

func (r *BuildingRepo) EnsureBuilding(ctx context.Context, coord domain.Coordinate) (domain.Building, error) {
	building, err := r.ensureOnce(ctx, coord)

	if isUniqueViolation(err) {
		return r.ensureOnce(ctx, coord)
	}

	return building, err
}

func (r *BuildingRepo) ensureOnce(ctx context.Context, coord domain.Coordinate) (domain.Building, error) {
	tx, err := r.pool.Begin(ctx)
	if err != nil {
		return domain.Building{}, fmt.Errorf("begin transaction: %w", err)
	}
	// rolls back on any early return; after a successful Commit it does nothing
	defer tx.Rollback(ctx)

	building, err := findNearby(ctx, tx, coord)
	if err == nil {
		return building, tx.Commit(ctx)
	}
	// pgx.ErrNoRows just means there is no building nearby
	if !errors.Is(err, pgx.ErrNoRows) {
		return domain.Building{}, fmt.Errorf("find building: %w", err)
	}

	building, err = insertBuilding(ctx, tx, coord)
	if err != nil {
		return domain.Building{}, err
	}

	// without this the worker never learns about the new building
	if err := enqueueJob(ctx, tx, building.ID); err != nil {
		return domain.Building{}, err
	}

	return building, tx.Commit(ctx)
}

func findNearby(ctx context.Context, tx pgx.Tx, coord domain.Coordinate) (domain.Building, error) {
	// ST_DWithin on geography measures metres, <-> orders by distance
	const query = `
		SELECT id, geo_lat, geo_lon, features_computed_at IS NOT NULL
		FROM buildings
		WHERE ST_DWithin(
			geom::geography,
			ST_SetSRID(ST_MakePoint($2, $1), 4326)::geography,
			$3
		)
		ORDER BY geom <-> ST_SetSRID(ST_MakePoint($2, $1), 4326)
		LIMIT 1`

	var b domain.Building
	err := tx.QueryRow(ctx, query, coord.Lat, coord.Lon, searchRadiusMeters).
		Scan(&b.ID, &b.Coord.Lat, &b.Coord.Lon, &b.FeaturesReady)

	return b, err
}

func insertBuilding(ctx context.Context, tx pgx.Tx, coord domain.Coordinate) (domain.Building, error) {
	// note the argument order: ST_MakePoint(longitude, latitude)
	const query = `
		INSERT INTO buildings (geo_lat, geo_lon, geom)
		VALUES ($1, $2, ST_SetSRID(ST_MakePoint($2, $1), 4326))
		RETURNING id`

	var id int64
	if err := tx.QueryRow(ctx, query, coord.Lat, coord.Lon).Scan(&id); err != nil {
		return domain.Building{}, fmt.Errorf("insert building: %w", err)
	}

	return domain.Building{ID: id, Coord: coord, FeaturesReady: false}, nil
}

func enqueueJob(ctx context.Context, tx pgx.Tx, buildingID int64) error {
	// ON CONFLICT DO NOTHING: a job for this building may already be queued
	const query = `
		INSERT INTO feature_jobs (building_id)
		VALUES ($1)
		ON CONFLICT (building_id) DO NOTHING`

	if _, err := tx.Exec(ctx, query, buildingID); err != nil {
		return fmt.Errorf("enqueue feature job: %w", err)
	}

	return nil
}

func isUniqueViolation(err error) bool {
	// errors.As unwraps the %w chain and checks the error type
	var pgErr *pgconn.PgError
	return errors.As(err, &pgErr) && pgErr.Code == uniqueViolation
}
