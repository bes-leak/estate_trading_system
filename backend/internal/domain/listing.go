package domain

import (
	"context"

	"github.com/bes-leak/estate_trading_system/backend/internal/estate_filters"
)

type Listing struct {
	ID     int64
	Price  int64
	Area   float32
	Rooms  int16
	Level  int16
	Levels int16
	Coord  Coordinate
}

type Page struct {
	Limit  int
	Offset int
}

type ListingRepository interface {
	Find(ctx context.Context, criterion estate_filters.Criterion, page Page) ([]Listing, error)
}
