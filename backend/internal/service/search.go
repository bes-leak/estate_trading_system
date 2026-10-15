package service

import (
	"github.com/bes-leak/estate_trading_system/backend/internal/estate_filters"
)

type SearchRequest struct {
	PriceMin *int64
	PriceMax *int64
	Rooms    []int16
	AreaMin  *float32
	AreaMax  *float32

	BBox         *BBox
	MetroWithin  *float64
	ExcludeFirst bool
}

type BBox struct {
	MinLat, MinLon, MaxLat, MaxLon float64
}

func BuildCriterion(request SearchRequest) estate_filters.Criterion {
	parts := []estate_filters.Criterion{}

	if request.PriceMin != nil || request.PriceMax != nil {
		filter := estate_filters.PriceFilter{Min: 0, Max: maxPrice}
		if request.PriceMin != nil {
			filter.Min = *request.PriceMin
		}
		if request.PriceMax != nil {
			filter.Max = *request.PriceMax
		}
		parts = append(parts, filter)
	}

	if len(request.Rooms) > 0 {
		parts = append(parts, estate_filters.RoomsIn{Rooms: request.Rooms})
	}

	if request.BBox != nil {
		parts = append(parts, estate_filters.InBbox{
			MinLat: request.BBox.MinLat, MinLon: request.BBox.MinLon,
			MaxLat: request.BBox.MaxLat, MaxLon: request.BBox.MaxLon,
		})
	}

	if request.MetroWithin != nil {
		parts = append(parts, estate_filters.MetroWithin{Meters: *request.MetroWithin})
	}

	if request.ExcludeFirst {
		parts = append(parts, estate_filters.NotFirstFloor{})
	}

	return estate_filters.And{Children: parts}
}

const maxPrice = 1_000_000_000
