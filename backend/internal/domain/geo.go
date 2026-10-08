package domain

import (
	"context"
	"errors"
)

type Coordinate struct {
	Lat float64
	Lon float64
}

type Geocoder interface {
	Geocode(ctx context.Context, address string) (Coordinate, error)
}

var ErrAddressNotFound = errors.New("address not found")
