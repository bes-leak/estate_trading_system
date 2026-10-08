package domain

import "context"

type Building struct {
	ID            int64
	Coord         Coordinate
	FeaturesReady bool
}

type BuildingRepository interface {
	EnsureBuilding(ctx context.Context, coord Coordinate) (Building, error)
}
